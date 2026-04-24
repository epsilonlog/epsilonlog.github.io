---
title: "PostgreSQL Write-Ahead Log (WAL) and Change Data Capture: A Deep Technical Analysis"
date: 2026-04-24 12:00:00 +0000
categories: [Data Engineering, Database Architecture]
tags: [postgresql, cdc, wal, mysql, database]
description: A comprehensive, deep-dive guide into the internal mechanics of Change Data Capture using PostgreSQL's WAL, including architecture, capacity planning, and comparisons with MySQL.
math: true
---

## 1. Introduction: The Problem CDC Solves {#introduction}

Change Data Capture (CDC) is the process of identifying and capturing changes made to data in a database, then delivering those changes in real-time to downstream systems. In 2026, CDC has securely cemented itself as the backbone of modern data architectures. It is the invisible engine enabling:

- **Real-time analytics** without impacting OLTP database performance.
- **Event-driven microservices** that instantly react to state changes.
- **Data synchronization** across heterogeneous systems (like syncing Postgres to Elasticsearch).
- **Audit trails** for stringent compliance and debugging.

The fundamental engineering challenge has always been: **How do we capture every single change without degrading the primary database's performance?**

Traditional approaches leaned on polling (e.g., executing `SELECT * FROM table WHERE updated_at > last_check`). However, polling has severe limitations at scale. For example, if you have $100$ tables polled every second with a $10$ ms query time, you would easily consume $1$ full CPU core just for the overhead of change detection, making polling entirely impractical for high-throughput environments.

**Enter Write-Ahead Log (WAL) based CDC**: Instead of aggressively querying live tables and fighting for locks, we intercept the transaction log that PostgreSQL already maintains for its own crash recovery. By tapping into this stream, we get a highly efficient, real-time feed of database events.

---

## 2. Write-Ahead Log (WAL): The Foundation {#wal-foundation}

To understand CDC, you must first understand the heartbeat of PostgreSQL: the WAL. 

### 2.1 What is WAL?

The Write-Ahead Log is PostgreSQL's primary durability mechanism, ensuring that ACID properties are maintained. The core rule is simple: before any data page is modified in the main memory (shared buffers), the change must first be securely written to the WAL on disk. This sequential log ensures:

1. **Crash Recovery**: If the server loses power, PostgreSQL replays the WAL upon restart to restore the database to a consistent state.
2. **Replication**: Standby servers constantly stream and apply WAL records to stay perfectly synchronized with the primary.
3. **Point-in-Time Recovery (PITR)**: Archived WAL files allow administrators to "time-travel" and restore a database to any specific second in the past.

### 2.2 WAL as a Write Buffer: Performance Optimization

Beyond durability, the WAL serves as a **sequential write buffer** that dramatically improves database write performance. If databases wrote directly to data files, the disk heads would constantly seek different sectors, causing massive latency.

**Without WAL (Direct Writes):**
Application $\rightarrow$ Random disk writes to scattered data pages (Extremely slow, high I/O wait)

**With WAL (Buffered Writes):**
Application $\rightarrow$ Sequential WAL writes (Fast) $\rightarrow$ Background writer process $\rightarrow$ Batch writes to data pages.

**Why this matters for CDC**:
- **Sequential writes** are $10-100 \times$ faster than random writes on traditional storage.
- **Batching** allows PostgreSQL to group multiple changes to the same page before syncing to disk.
- Because WAL is sequential, reading it for CDC is also incredibly fast and sequential, causing almost zero disk contention with primary OLTP queries.

**Default WAL buffer configuration**:

```python
# postgresql.conf
wal_buffers = -1  # Auto-sized (typically 1/32 of shared_buffers)
# Default: ~$512$ KB to $16$ MB depending on shared_buffers
```

> **Failure Scenario #1: Insufficient max_wal_senders**
> 
> **Symptom**: "FATAL: number of requested standby connections exceeds max_wal_senders"
> **Cause**: More replication connections (or CDC consumers) than the configured limit.
> **Impact**: New replicas or CDC connectors simply cannot connect.
> **Solution**:  
> `ALTER SYSTEM SET max_wal_senders = 20;`  
> `SELECT pg_reload_conf();`
{: .prompt-danger }

### 2.3 WAL Record Structure

Under the hood, the WAL is a binary stream. Each WAL record contains a strict header followed by the actual payload:

```c
typedef struct XLogRecord {
    uint32      xl_tot_len;     /* Total length of record */
    TransactionId xl_xid;       /* Transaction ID */
    XLogRecPtr  xl_prev;        /* Pointer to previous record */
    uint8       xl_info;        /* Flag bits */
    RmgrId      xl_rmid;        /* Resource manager ID */
    /* Followed by actual data */
} XLogRecord;
```

**Key insight**: Because WAL records are **append-only** and structurally **sequential**, they are structurally perfect for event streaming platforms like Apache Kafka.

### 2.4 WAL Levels

PostgreSQL does not log everything by default. The amount of detail written to the WAL is dictated by the `wal_level`. PostgreSQL supports three distinct WAL levels:

| Level | Description | Use Case | CPU/I/O Overhead |
|-------|-------------|----------|----------|
| `minimal` | Only basic crash recovery info | Single-server, no replication | Lowest (~$2\%$) |
| `replica` | Includes physical block changes | Streaming replication | Medium (~$5\%$) |
| `logical` | Adds logical row state data | CDC, logical replication | Highest (~$10-15\%$) |

For CDC to function, the database must write the actual row data, meaning we absolutely require `wal_level = logical`:

```python
# postgresql.conf
wal_level = logical
max_wal_senders = 10          # Number of concurrent WAL sender processes
max_replication_slots = 10    # Number of replication slots
```

### 2.5 Logical Decoding: From Binary to Structured Changes

Raw WAL is heavily optimized for machine reading—it is binary and refers to physical disk blocks. CDC requires a translator. **Logical decoding** is the internal PostgreSQL process that transforms these dense binary WAL records into structured, human-readable change events:

```text
Binary WAL Record (Physical blocks):
0x52 0x00 0x00 0x00 0x01 0x00 0x00 0x00 ...
```
$\downarrow$ Logical Decoding Plugin (e.g., pgoutput, wal2json) $\downarrow$
```json
{
  "operation": "INSERT",
  "schema": "public",
  "table": "users",
  "columns": {
    "id": 123,
    "name": "Alice",
    "email": "alice@example.com"
  }
}
```

---

## 3. TCP Connection Architecture for WAL Streaming {#tcp-architecture}

Once the WAL is decoded, it needs to leave the database. PostgreSQL uses a custom replication protocol over TCP to stream these changes continuously, pushing them to the CDC client rather than waiting for the client to poll. 

Because network partitions happen, configuring the TCP connection for high reliability is critical so the database doesn't hold onto WAL files for a dead client.

```python
# Python example using psycopg2 for logical replication
import psycopg2

conn = psycopg2.connect(
    host="postgres-primary",
    port=5432,
    user="replication_user",
    password="secure_password",
    dbname="postgres",
    # Replication-specific parameters flag this as a streaming connection
    replication="database",  
    
    # TCP keepalive is vital to detect silent connection failures
    keepalives=1,
    keepalives_idle=30,      # Start sending keepalives after 30s idle
    keepalives_interval=10,  # Send a keepalive packet every 10s
    keepalives_count=5       # Drop connection after 5 failed attempts
)
```

**TCP Keepalive Math**: Time to detect a silent connection failure = $30 + (5 \times 10) = 80$ seconds.

> **Failure Scenario #2: Network Partition**
> 
> **Symptom**: Replication slot shows increasing lag, but no errors in the logs.
> **Cause**: A network partition occurred without a clean TCP connection drop (e.g., a firewall dropped state).
> **Impact**: The database thinks the client is just slow, so WAL accumulates on the primary, potentially leading to a disk full event.
> **Solution**: Keepalive timers ensure the DB proactively kills the ghost connection.
{: .prompt-warning }

---

## 4. Logical vs Physical Replication: Deep Comparison {#replication-comparison}

It is crucial to differentiate between the two main types of replication PostgreSQL offers, as they serve different operational purposes.

### 4.1 Physical Replication

**Characteristics**:
- Creates a byte-for-byte identical physical replica (standby).
- Cannot filter tables or columns; it replicates the whole cluster.
- Lowest latency ($<10$ ms typical).
- Minimal CPU overhead (~$2-5\%$).

### 4.2 Logical Replication (Used for CDC)

**Characteristics**:
- Transmits row-level data changes in a logical format.
- Can selectively filter specific tables, columns, and even rows.
- Slightly higher latency ($50-200$ ms typical) due to the decoding overhead.
- Higher CPU overhead (~$10-20\%$) because decoding binary blocks back into row structures requires compute.

### 4.3 PostgreSQL 16+: Logical Decoding on Physical Standbys

Historically, CDC tools had to connect directly to the primary database, adding load. PostgreSQL 16 introduced a game-changer: the ability to run logical decoding on a physical standby. This offloads the $10-20\%$ CPU decoding penalty from your primary OLTP database to a read-replica.

```sql
-- On primary: Enable logical replication base
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 10;

-- On standby: Enable hot_standby_feedback so primary doesn't vacuum needed rows
ALTER SYSTEM SET hot_standby_feedback = on;
ALTER SYSTEM SET max_replication_slots = 10;

-- Create slot on standby (PostgreSQL 16+)
SELECT pg_create_logical_replication_slot(
    'standby_cdc_slot',
    'pgoutput'
);
```

---

## 5. Replication Slots: The Coordination Mechanism {#replication-slots}

If a CDC consumer disconnects, how does PostgreSQL know not to delete the WAL files the consumer hasn't read yet? **Replication slots**.

A replication slot represents a permanent cursor in the WAL stream. It ensures that the database will rigidly retain WAL segments until the specific consumer attached to that slot acknowledges it has safely processed them.

### 5.1 Slot Architecture and LSN

PostgreSQL addresses exact positions in the WAL using a Log Sequence Number (LSN). You can track exactly how far behind a CDC consumer is by comparing the server's current LSN to the slot's confirmed LSN.

```sql
-- Calculate WAL lag in bytes
SELECT pg_wal_lsn_diff(
    pg_current_wal_lsn(),           -- Current database write position
    confirmed_flush_lsn              -- Last position acknowledged by CDC slot
) AS lag_bytes
FROM pg_replication_slots
WHERE slot_name = 'debezium_slot';
```

> **Failure Scenario #5: Inactive Slot Disk Exhaustion**
> 
> **Symptom**: "PANIC: could not write to file pg_wal/...: No space left on device"  
> **Cause**: A CDC consumer crashed, leaving an inactive replication slot. PostgreSQL faithfully retained all WAL for hours/days until the disk hit $100\%$.  
> **Prevention**: Always set maximum WAL retention (PostgreSQL 13+) to protect the primary: `ALTER SYSTEM SET max_slot_wal_keep_size = '100GB';`
{: .prompt-danger }

### 5.2 Slot Monitoring and Alerting

To prevent catastrophic disk full events, aggressive monitoring is required. **Recommended alert thresholds**:
- **WARNING**: Slot lag > $10$ GB
- **CRITICAL**: Slot lag > $50$ GB

---

## 6. MySQL's Alternative Approach: Binlog Architecture {#mysql-comparison}

It is useful to contrast Postgres with MySQL. MySQL uses a fundamentally different architecture for replication and CDC. Instead of a write-ahead log (which logs before a write), MySQL uses a **binary log (binlog)** that records changes **after** they are committed to the storage engine. 

### 6.1 Binlog Formats

MySQL's binlog supports different formats. For CDC to capture exact data payloads, you must configure it to use ROW format, ensuring before-and-after states are fully captured.

```sql
SET GLOBAL binlog_format = 'ROW';
SET GLOBAL binlog_row_image = 'FULL';  -- Include before/after images in the event
```

### 6.2 No Replication Slots: The GTID Solution

A massive architectural difference: MySQL doesn't have replication slots. It will happily delete binlogs based on a timer, regardless of whether your CDC consumer has read them. Instead of slots, it uses **GTID (Global Transaction Identifiers)** to allow clients to track their own position.

> **Tip**
> Unlike PostgreSQL slots that automatically retain WAL, MySQL requires manual binlog retention configuration. Set `binlog_expire_logs_seconds` conservatively (e.g., $3-7$ days) to prevent CDC failures during long weekend outages.
{: .prompt-info }

---

## 7. Capacity Planning {#capacity-planning}

Day-two operations require math. Implementing CDC changes your network and storage requirements significantly. Here is how to model the load.

### 7.1 WAL/Binlog Generation Rate

**PostgreSQL WAL:**

$$
\text{WAL rate (MB/s)} = \text{Write rate (rows/s)} \times \text{Avg row size (bytes)} \times \frac{1.3}{1024^2}
$$

*The $1.3$ multiplier accounts for WAL overhead (headers, block alignment, TOAST pointers).*

**MySQL Binlog:**

$$
\text{Binlog rate (MB/s)} = \text{Write rate (rows/s)} \times \text{Avg row size (bytes)} \times \frac{1.2}{1024^2}
$$

*The $1.2$ multiplier accounts for binlog event headers and metadata.*

**Example:**
- $10,000$ writes/sec
- Average row size: $500$ bytes
- PostgreSQL: $10000 \times 500 \times \frac{1.3}{1024^2} = 6.2$ MB/s
- MySQL: $10000 \times 500 \times \frac{1.2}{1024^2} = 5.7$ MB/s

### 7.2 Network Bandwidth

You must provision enough network bandwidth between the database and the CDC platform (e.g., Debezium/Kafka).

**Physical Replication (PostgreSQL Standby):**

$$
\text{Bandwidth (Mbps)} = \text{WAL rate (MB/s)} \times 8
$$

**Logical Replication (PostgreSQL/MySQL CDC):**

$$
\text{Bandwidth (Mbps)} = \text{WAL/Binlog rate (MB/s)} \times 0.6 \times 8
$$

*The $0.6$ factor accounts for logical filtering and network compression (a typical $40\%$ reduction).*

### 7.3 Disk Space Calculation

How much extra disk space do you need to survive a CDC consumer outage?

**PostgreSQL WAL:**

$$
\text{WAL disk (GB)} = \frac{\text{WAL rate (MB/s)} \times \text{Max downtime (seconds)}}{1024} \times 1.2
$$

*The $1.2$ adds a $20\%$ safety buffer.*

**MySQL Binlog:**

$$
\text{Binlog disk (GB)} = \frac{\text{Binlog rate (MB/s)} \times \text{Retention period (seconds)}}{1024}
$$

**Example:**
- WAL rate: $6.2$ MB/s
- Max acceptable downtime: $4$ hours ($14,400$ seconds)
- Required WAL Buffer: $6.2 \times \frac{14400}{1024} \times 1.2 = 104$ GB.

### 7.4 CPU Overhead Expectations

**PostgreSQL Logical Decoding:**
- Baseline: $5-10\%$ CPU on primary.
- Factors increasing overhead: Large transactions ($>10,000$ rows), complex data types (JSON, arrays), high write rate ($>50,000$ rows/sec), and multiple logical slots (add $3-5\%$ per slot).

**MySQL Binlog Writing:**
- Baseline: $2-5\%$ CPU on primary.
- Factors increasing overhead: `binlog_row_image = FULL`, `sync_binlog = 1`, or statement-based logging on complex queries.

**CDC Connector Processing (e.g., Debezium):**
- Typically requires $0.5 - 1.0$ CPU core per $10,000$ events/sec.

---

## 8. Production Checklist {#production-checklist}

### 8.1 PostgreSQL CDC Setup

**1. Enable Logical Replication:**
```ini
# postgresql.conf
wal_level = logical
max_replication_slots = 10
max_wal_senders = 10
```

**2. Create a Dedicated Replication User:**
```sql
CREATE ROLE cdc_user WITH REPLICATION LOGIN PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO cdc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO cdc_user;
```

**3. Define the Publication (What to stream):**
```sql
CREATE PUBLICATION cdc_pub FOR ALL TABLES;
-- Or be selective to save bandwidth:
CREATE PUBLICATION cdc_pub FOR TABLE users, orders;
```

**4. Create the Replication Slot:**
```sql
SELECT pg_create_logical_replication_slot('debezium_slot', 'pgoutput');
```

**5. Configure Network Access (pg_hba.conf):**
```text
host replication cdc_user 10.0.0.0/8 scram-sha-256
```

### 8.2 MySQL CDC Setup

**1. Enable and Configure Binlog:**
```ini
# my.cnf
server-id = 1
log_bin = /var/lib/mysql/mysql-bin
binlog_format = ROW
binlog_row_image = FULL
gtid_mode = ON
enforce_gtid_consistency = ON
binlog_expire_logs_seconds = 259200  # 3 days guaranteed retention
```

**2. Create Replication User:**
```sql
CREATE USER 'cdc_user'@'%' IDENTIFIED BY 'secure_password';
GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'cdc_user'@'%';
FLUSH PRIVILEGES;
```

---

## 9. Essential Monitoring Queries {#monitoring-queries}

Arm your observability platform with these queries to detect CDC failure states before they cause production outages.

### 9.1 PostgreSQL Diagnostics

**Replication Lag (Live View):**
```sql
SELECT
    slot_name,
    client_addr,
    state,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn)) AS send_lag,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn)) AS write_lag,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn)) AS flush_lag,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS replay_lag
FROM pg_stat_replication;
```

**Find Dangerous Inactive Slots:**
```sql
SELECT
    slot_name,
    active,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained_wal,
    CASE WHEN active THEN NULL
         ELSE EXTRACT(EPOCH FROM (now() - (SELECT max(last_msg_send_time) FROM pg_stat_replication WHERE slot_name = s.slot_name)))
    END AS inactive_seconds
FROM pg_replication_slots s
WHERE slot_type = 'logical'
ORDER BY pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) DESC;
```

### 9.2 MySQL Diagnostics

**Binlog Generation Rate / Disk Usage:**
```sql
SELECT VARIABLE_VALUE AS binlog_size_mb
FROM performance_schema.global_status
WHERE VARIABLE_NAME = 'Binlog_cache_disk_use';
```

**Total Active Binlog Size on Disk:**
```sql
SELECT CONCAT(ROUND(SUM(file_size) / 1024 / 1024, 2), ' MB') AS total_binlog_size
FROM (
    SELECT file_size
    FROM information_schema.files
    WHERE file_name LIKE '%binlog%'
) AS binlog_files;
```

---

## 10. Comparison Table: PostgreSQL vs MySQL CDC {#comparison}

To summarize, here is how the two database giants handle Change Data Capture at an architectural level.

| Aspect | PostgreSQL | MySQL |
|--------|-----------|-------|
| **Mechanism** | WAL (write-ahead log) | Binlog (after-commit log) |
| **Position Tracking** | Replication slots (server-side tracking) | GTID (client-side tracking) |
| **Automatic Retention** | Yes (via active slots) | No (manual `binlog_expire_logs_seconds`) |
| **Output Format** | Logical decoding plugins (`pgoutput`, `wal2json`) | ROW / STATEMENT / MIXED |
| **Durability** | WAL required for ACID compliance | Binlog technically optional (`sync_binlog=0`) |
| **Filtering** | Table / column / row natively (via publications) | Handled downstream (Debezium-side) |
| **Schema Changes (DDL)**| Not natively replicated in logical streams | Captured directly in binlog |
| **Snapshot Locking** | `REPEATABLE READ` (no restrictive locks) | Requires table locks during initial snapshot |
| **Failover** | Slot recreation needed (PG16), auto-sync (PG17+) | GTID auto-continues seamlessly |
| **Primary Overhead** | ~$5-10\%$ CPU (due to logical decoding) | ~$2-5\%$ CPU (just writing binlog) |
| **Disk Exhaustion Risk**| High (Inactive slots indefinitely hold WAL) | Low (Time-based purge protects disk) |
| **Typical Latency** | $10-50$ ms | $5-30$ ms |

### The Final Verdict: Key Takeaways

1. **Protect your disk**: Always monitor inactive replication slots in PostgreSQL and binlog disk usage in MySQL. An unmonitored CDC setup is a ticking time bomb for database stability.
2. **Alert aggressively**: Set up hard alerting for lag thresholds ($>1$ GB for PostgreSQL, $>1$ hour for MySQL).
3. **Offload when possible**: Use PostgreSQL 16+ standby CDC for production architectures to eliminate the decoding CPU penalty on your primary OLTP node.
4. **Capacity Plan**: Always plan disk capacity for $2-3 \times$ your expected CDC outage retention period. Calculate using the formulas provided.