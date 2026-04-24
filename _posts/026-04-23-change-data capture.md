---
title: "PostgreSQL Write-Ahead Log (WAL) and Change Data Capture: A Deep Technical Analysis"
date: 2026-04-23 12:00:00 +0000
categories: [Data Engineering, Database Architecture]
tags: [postgresql, cdc, wal, mysql, database]
description: 
math: true
---



## 1. Introduction: The Problem CDC Solves {#introduction}

Change Data Capture (CDC) is the process of identifying and capturing changes made to data in a database, then delivering those changes in real-time to downstream systems. In 2026, CDC has become the backbone of modern data architectures, enabling:

- **Real-time analytics** without impacting OLTP performance
- **Event-driven microservices** that react to data changes
- **Data synchronization** across heterogeneous systems
- **Audit trails** for compliance and debugging

The fundamental challenge: **How do we capture every change without degrading database performance?**

Traditional approaches like polling (`SELECT * FROM table WHERE updated_at > last_check`) have severe limitations. For $100$ tables polled every second with $10$ms query time, you'd consume $1$ full CPU core just for change detection, making polling impractical at scale.

**Enter Write-Ahead Log (WAL) based CDC**: Instead of querying tables, we read the transaction log that PostgreSQL already maintains for crash recovery.

---

## 2. Write-Ahead Log (WAL): The Foundation {#wal-foundation}

### 2.1 What is WAL?

The Write-Ahead Log is PostgreSQL's durability mechanism. Before any data page is modified in memory, the change is first written to WAL. This ensures:

1. **Crash Recovery**: After a crash, PostgreSQL replays WAL to restore consistent state
2. **Replication**: Standby servers apply WAL to stay synchronized
3. **Point-in-Time Recovery (PITR)**: WAL archives enable time-travel

### 2.2 WAL as a Write Buffer: Performance Optimization

WAL serves as a **sequential write buffer** that dramatically improves write performance:
```text
Without WAL (Direct Writes):
Application → Random disk writes to data pages
(slow, requires seeking)

With WAL (Buffered Writes):
Application → Sequential WAL writes (fast)
↓
Background process
↓
Batch writes to data pages

**Why this matters**:

- **Sequential writes** are $10-100\times$ faster than random writes on traditional disks
- **Batching** allows PostgreSQL to group multiple changes to the same page
- **Delayed writes** enable the OS page cache to optimize I/O

**Default WAL buffer configuration**:

sql
-- postgresql.conf
wal_buffers = -1  -- Auto-sized (typically 1/32 of shared_buffers)
-- Default: ~512KB to 16MB depending on shared_buffers

> **Failure Scenario #1: Insufficient max_wal_senders**
> 
> **Symptom**: "FATAL: number of requested standby connections exceeds max_wal_senders"
> **Cause**: More replication connections than configured limit
> **Impact**: New replicas or CDC connectors cannot connect
> **Solution**:  
> `ALTER SYSTEM SET max_wal_senders = 20;`  
> `SELECT pg_reload_conf();`
{: .prompt-danger }

### 2.3 WAL Record Structure

Each WAL record contains:

c
typedef struct XLogRecord {
uint32      xl_tot_len;     /* Total length of record */
TransactionId xl_xid;       /* Transaction ID */
XLogRecPtr  xl_prev;        /* Pointer to previous record */
uint8       xl_info;        /* Flag bits */
RmgrId      xl_rmid;        /* Resource manager ID */
/* Followed by actual data */
} XLogRecord;

**Key insight**: WAL records are **append-only** and **sequential**, making them extremely efficient to write and read.

### 2.4 WAL Levels

PostgreSQL supports three WAL levels:

| Level | Description | Use Case | Overhead |
|-------|-------------|----------|----------|
| `minimal` | Only crash recovery info | Single-server, no replication | Lowest (~$2\%$) |
| `replica` | Physical replication data | Streaming replication | Medium (~$5\%$) |
| `logical` | Logical decoding data | CDC, logical replication | Highest (~$10-15\%$) |

For CDC, we need `wal_level = logical`:

sql
-- postgresql.conf
wal_level = logical
max_wal_senders = 10          -- Number of concurrent WAL sender processes
max_replication_slots = 10    -- Number of replication slots

### 2.5 Logical Decoding: From Binary to Structured Changes

Logical decoding transforms binary WAL records into structured change events:

json
Binary WAL Record:
0x52 0x00 0x00 0x00 0x01 0x00 0x00 0x00 ...

↓ Logical Decoding ↓

Structured Change Event:
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

---

## 3. TCP Connection Architecture for WAL Streaming {#tcp-architecture}

### 3.1 The Replication Protocol

PostgreSQL uses a custom protocol over TCP for WAL streaming.

### 3.2 TCP Connection Parameters

The replication connection uses specific TCP settings for reliability:

python
# Python example using psycopg2
import psycopg2

conn = psycopg2.connect(
host="postgres-primary",
port=5432,
user="replication_user",
password="secure_password",
dbname="postgres",
# Replication-specific parameters
replication="database",  # or "true" for physical replication
# TCP keepalive to detect connection failures
keepalives=1,
keepalives_idle=30,      # Start keepalive after 30s idle
keepalives_interval=10,  # Send keepalive every 10s
keepalives_count=5       # Drop connection after 5 failed keepalives
)

**TCP Keepalive**: Time to detect connection failure = $30 + (5 \times 10) = 80$ seconds.

> **Failure Scenario #2: Network Partition**
> 
> **Symptom**: Replication slot shows increasing lag, but no errors  
> **Cause**: Network partition without TCP connection drop  
> **Impact**: WAL accumulates on primary, potential disk full  
> **Solution**: Reduce keepalive timers for faster detection
{: .prompt-warning }

---

## 4. Logical vs Physical Replication: Deep Comparison {#replication-comparison}

### 4.1 Physical Replication

**Characteristics**:
- Byte-for-byte identical replica
- Cannot filter tables or columns
- Lowest latency ($<10$ms typical)
- Minimal CPU overhead (~$2-5\%$)

### 4.2 Logical Replication

**Characteristics**:
- Row-level changes in logical format
- Can filter tables, columns, and rows
- Higher latency ($50-200$ms typical)
- Higher CPU overhead (~$10-20\%$)

### 4.3 PostgreSQL 16+: Logical Decoding on Physical Standbys

PostgreSQL 16 introduced the best of both worlds: Physical replication for the standby, but with logical decoding capability.

sql
-- On primary: Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 10;

-- On standby: Enable hot_standby_feedback and slots
ALTER SYSTEM SET hot_standby_feedback = on;
ALTER SYSTEM SET max_replication_slots = 10;

-- Create slot on standby (PostgreSQL 16+)
SELECT pg_create_logical_replication_slot(
'standby_cdc_slot',
'pgoutput'
);

---

## 5. Replication Slots: The Coordination Mechanism {#replication-slots}

Without replication slots, PostgreSQL faces a dilemma regarding when to delete WAL segments. **Replication slots** track each replica's progress and prevent WAL deletion until all replicas have consumed it.

### 5.1 Slot Architecture and LSN

LSN is PostgreSQL's way of addressing WAL positions:

sql
-- Calculate WAL lag in bytes
SELECT pg_wal_lsn_diff(
pg_current_wal_lsn(),           -- Current position
confirmed_flush_lsn              -- Slot position
) AS lag_bytes
FROM pg_replication_slots
WHERE slot_name = 'debezium_slot';

> **Failure Scenario #5: Inactive Slot Disk Exhaustion**
> 
> **Symptom**: "PANIC: could not write to file pg_wal/...: No space left on device"  
> **Cause**: Inactive replication slot retaining WAL for hours/days  
> **Prevention**: Set maximum WAL retention (PostgreSQL 13+) `ALTER SYSTEM SET max_slot_wal_keep_size = '100GB';`
{: .prompt-danger }

### 5.2 Slot Monitoring and Alerting

**Recommended alert thresholds**:

- **WARNING**: Slot lag > $10$ GB
- **CRITICAL**: Slot lag > $50$ GB

---

## 6. MySQL's Alternative Approach: Binlog Architecture {#mysql-comparison}

MySQL uses a fundamentally different architecture for replication and CDC. Instead of a write-ahead log, MySQL has a **binary log (binlog)** that records changes **after** they're committed to storage.

### 6.1 Binlog Formats

For CDC, always use ROW format:

sql
SET GLOBAL binlog_format = 'ROW';
SET GLOBAL binlog_row_image = 'FULL';  -- Include before/after images

### 6.2 No Replication Slots: The GTID Solution

MySQL doesn't have replication slots. Instead, it uses **GTID (Global Transaction Identifier)**.

> **Tip**
> Unlike PostgreSQL slots that automatically retain WAL, MySQL requires manual binlog retention configuration. Set `binlog_expire_logs_seconds` conservatively to prevent CDC failures.
{: .prompt-info }

## Capacity Planning

### WAL/Binlog Generation Rate

**PostgreSQL WAL:**

$$
\text{WAL rate (MB/s)} = \text{Write rate (rows/s)} \times \text{Avg row size (bytes)} \times 1.3 / 1024^2
$$

The $1.3$ multiplier accounts for WAL overhead (headers, alignment, TOAST).

**MySQL Binlog:**

$$
\text{Binlog rate (MB/s)} = \text{Write rate (rows/s)} \times \text{Avg row size (bytes)} \times 1.2 / 1024^2
$$

The $1.2$ multiplier accounts for binlog event headers and metadata.

**Example:**
- 10,000 writes/sec
- Average row size: 500 bytes
- PostgreSQL: $10000 \times 500 \times 1.3 / 1024^2 = 6.2$ MB/s
- MySQL: $10000 \times 500 \times 1.2 / 1024^2 = 5.7$ MB/s

### Network Bandwidth

**Physical Replication (PostgreSQL):**

$$
\text{Bandwidth (Mbps)} = \text{WAL rate (MB/s)} \times 8
$$

**Logical Replication (PostgreSQL/MySQL CDC):**

$$
\text{Bandwidth (Mbps)} = \text{WAL/Binlog rate (MB/s)} \times 0.6 \times 8
$$

The $0.6$ factor accounts for filtering and compression (typical 40% reduction).

### Disk Space Calculation

**PostgreSQL WAL:**

$$
\text{WAL disk (GB)} = \text{WAL rate (MB/s)} \times \text{Max downtime (seconds)} / 1024
$$

Add 20% buffer for safety.

**MySQL Binlog:**

$$
\text{Binlog disk (GB)} = \text{Binlog rate (MB/s)} \times \text{Retention period (seconds)} / 1024
$$

**Example:**
- WAL rate: 6.2 MB/s
- Max acceptable downtime: 4 hours (14,400 seconds)
- Required: $6.2 \times 14400 / 1024 \times 1.2 = 104$ GB

### CPU Overhead

**PostgreSQL Logical Decoding:**
- Baseline: 5-10% CPU on primary
- Factors increasing overhead:
  - Large transactions (>10,000 rows)
  - Complex data types (JSON, arrays)
  - High write rate (>50,000 rows/sec)
  - Multiple logical slots (add 3-5% per slot)

**MySQL Binlog Writing:**
- Baseline: 2-5% CPU on primary
- Factors increasing overhead:
  - `binlog_row_image = FULL` (vs `MINIMAL`)
  - `sync_binlog = 1` (vs `0`)
  - Statement-based logging with complex queries

**CDC Connector Processing:**
- Debezium/consumer: 0.5-1 CPU core per 10,000 events/sec
- Kafka: 2-4 CPU cores for broker handling CDC topics

## Production Checklist

### PostgreSQL CDC Setup

**1. Enable Logical Replication:**
```sql
-- postgresql.conf
wal_level = logical
max_replication_slots = 10
max_wal_senders = 10

**2. Create Replication User:**

sql
CREATE ROLE cdc_user WITH REPLICATION LOGIN PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO cdc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO cdc_user;

**3. Create Publication:**

sql
CREATE PUBLICATION cdc_pub FOR ALL TABLES;
-- Or selective:
CREATE PUBLICATION cdc_pub FOR TABLE users, orders;

**4. Create Replication Slot:**

sql
SELECT pg_create_logical_replication_slot('debezium_slot', 'pgoutput');

**5. Configure pg_hba.conf:**


host replication cdc_user 10.0.0.0/8 scram-sha-256

**6. Set Up Monitoring:**

sql
-- Create monitoring view
CREATE VIEW cdc_health AS
SELECT
slot_name,
active,
pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS lag_size,
EXTRACT(EPOCH FROM (now() - last_msg_send_time)) AS seconds_since_last_msg
FROM pg_replication_slots
WHERE slot_type = 'logical';

### MySQL CDC Setup

**1. Enable Binlog:**

ini
# my.cnf
server-id = 1
log_bin = /var/lib/mysql/mysql-bin
binlog_format = ROW
binlog_row_image = FULL
gtid_mode = ON
enforce_gtid_consistency = ON
binlog_expire_logs_seconds = 259200  # 3 days

**2. Create Replication User:**

sql
CREATE USER 'cdc_user'@'%' IDENTIFIED BY 'secure_password';
GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'cdc_user'@'%';
FLUSH PRIVILEGES;

**3. Verify Configuration:**

sql
SHOW VARIABLES LIKE 'binlog_format';
SHOW VARIABLES LIKE 'gtid_mode';
SHOW MASTER STATUS;

**4. Set Up Monitoring:**

sql
-- Create monitoring view
CREATE VIEW cdc_health AS
SELECT
VARIABLE_VALUE AS binlog_file
FROM performance_schema.global_status
WHERE VARIABLE_NAME = 'Binlog_cache_disk_use'
UNION ALL
SELECT
CONCAT(ROUND(SUM(data_length + index_length) / 1024 / 1024, 2), ' MB') AS total_size
FROM information_schema.tables;

## Monitoring Queries

### PostgreSQL

**Replication Lag:**

sql
SELECT
slot_name,
client_addr,
state,
pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn)) AS send_lag,
pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn)) AS write_lag,
pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn)) AS flush_lag,
pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS replay_lag
FROM pg_stat_replication;

**WAL Generation Rate:**

sql
SELECT
pg_size_pretty(
pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0')
) AS total_wal_generated,
pg_size_pretty(
pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0') / 
EXTRACT(EPOCH FROM (now() - pg_postmaster_start_time()))
) AS wal_rate_per_sec
FROM pg_stat_database
WHERE datname = current_database();

**Inactive Slots:**

sql
SELECT
slot_name,
slot_type,
database,
active,
pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained_wal,
CASE
WHEN active THEN NULL
ELSE EXTRACT(EPOCH FROM (now() - 
(SELECT max(last_msg_send_time) FROM pg_stat_replication WHERE slot_name = s.slot_name)
))
END AS inactive_seconds
FROM pg_replication_slots s
WHERE slot_type = 'logical'
ORDER BY pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) DESC;

**Active WAL Senders:**

sql
SELECT
pid,
usename,
application_name,
client_addr,
state,
sync_state,
pg_size_pretty(sent_lsn - '0/0'::pg_lsn) AS sent,
pg_size_pretty(write_lsn - '0/0'::pg_lsn) AS written,
pg_size_pretty(flush_lsn - '0/0'::pg_lsn) AS flushed,
backend_start,
EXTRACT(EPOCH FROM (now() - backend_start)) AS connection_age_seconds
FROM pg_stat_replication;

### MySQL

**Replication Lag (Replica):**

sql
SHOW SLAVE STATUS\G
-- Look for: Seconds_Behind_Master, Slave_IO_Running, Slave_SQL_Running

**Binlog Generation Rate:**

sql
SELECT
VARIABLE_VALUE AS binlog_size_mb
FROM performance_schema.global_status
WHERE VARIABLE_NAME = 'Binlog_cache_disk_use';

-- Calculate rate over time
SELECT
(@current_pos := CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(@@global.gtid_executed, ':', -1), '-', -1) AS UNSIGNED)) AS current_position,
@current_pos - @previous_pos AS transactions_per_interval
FROM (SELECT @previous_pos := 0) AS init;

**Binlog Files:**

sql
SHOW BINARY LOGS;

-- Total binlog size
SELECT
CONCAT(ROUND(SUM(file_size) / 1024 / 1024, 2), ' MB') AS total_binlog_size
FROM (
SELECT file_size
FROM information_schema.files
WHERE file_name LIKE '%binlog%'
) AS binlog_files;

**Active Replicas:**

sql
SHOW PROCESSLIST;
-- Look for: Command = 'Binlog Dump' or 'Binlog Dump GTID'

SELECT
ID,
USER,
HOST,
COMMAND,
TIME,
STATE
FROM information_schema.processlist
WHERE COMMAND LIKE 'Binlog Dump%';

## Comparison Table: PostgreSQL vs MySQL CDC

| Aspect | PostgreSQL | MySQL |
|--------|-----------|-------|
| **Mechanism** | WAL (write-ahead) | Binlog (after-commit) |
| **Position Tracking** | Replication slots (server-side) | GTID (client-side) |
| **Automatic Retention** | Yes (via slots) | No (manual `binlog_expire_logs_seconds`) |
| **Output Format** | Logical decoding plugins (`pgoutput`, `wal2json`) | ROW / STATEMENT / MIXED |
| **Durability** | WAL required for ACID | Binlog optional (`sync_binlog=0` allowed) |
| **Filtering** | Table / column / row (pgoutput) | Database / table (Debezium-side) |
| **Schema Changes** | Captured automatically | Requires separate history topic |
| **Snapshot Locking** | `REPEATABLE READ` (no locks) | Table locks during snapshot |
| **Failover** | Slot recreation needed (PG16), auto-sync (PG17+) | GTID auto-continues |
| **Primary Overhead** | 5-10% CPU (logical decoding) | 2-5% CPU (binlog writing) |
| **Disk Risk** | Inactive slots fill disk | Time-based purge (safer) |
| **Cross-version** | Yes (logical replication) | Yes (GTID-based) |
| **Typical Latency** | 10-50ms | 5-30ms |
| **Default Output** | Binary (requires plugin for JSON) | Binary (requires `mysqlbinlog` to read) |
| **DDL Replication** | Not supported natively | Captured in binlog |
| **TOAST / Large Values** | Handled via `REPLICA IDENTITY FULL` | `binlog_row_image = FULL` |
| **Max Consumers** | Limited by `max_wal_senders` | Limited by `max_connections` |

## Conclusion

Both PostgreSQL and MySQL offer robust CDC capabilities, but with different architectural philosophies:

**Choose PostgreSQL CDC when:**
- You need fine-grained filtering (column-level, row-level)
- Schema evolution tracking is critical
- You want server-side position management (replication slots)
- You're already using PostgreSQL and want native integration
- You can tolerate slightly higher primary overhead (5-10% CPU)
- PostgreSQL 16+ allows offloading CDC to standbys for production workloads

**Choose MySQL CDC when:**
- You need minimal primary overhead (2-5% CPU)
- GTID-based self-healing replication is preferred
- You want simpler retention management (time-based)
- Faster typical latency (5-30ms) is important
- You're comfortable with client-side position tracking
- DDL change capture is required

**Hybrid Approach:**
Many organizations run both databases and use a unified CDC platform (like Debezium) to normalize the differences, feeding into a common event streaming layer (Kafka, Pulsar).

**Key Takeaways:**
1. Always monitor inactive replication slots (PostgreSQL) and binlog disk usage (MySQL)
2. Set up alerting for lag thresholds (>1GB for PostgreSQL, >1 hour for MySQL)
3. Test failover scenarios regularly
4. Use PostgreSQL 16+ standby CDC for production to reduce primary load
5. Enable GTID in MySQL for easier failover and position tracking
6. Plan disk capacity for 2-3x your expected retention period
7. Implement circuit breakers for slow consumers to prevent backpressure

## Further Reading

**PostgreSQL:**
- [Logical Replication Documentation](https://www.postgresql.org/docs/current/logical-replication.html)
- [Replication Slots](https://www.postgresql.org/docs/current/warm-standby.html#STREAMING-REPLICATION-SLOTS)
- [PostgreSQL 16 Release Notes - Standby Logical Decoding](https://www.postgresql.org/docs/16/release-16.html)
- [WAL Internals](https://www.postgresql.org/docs/current/wal-internals.html)

**MySQL:**
- [Binary Log Documentation](https://dev.mysql.com/doc/refman/8.0/en/binary-log.html)
- [GTID Replication](https://dev.mysql.com/doc/refman/8.0/en/replication-gtids.html)
- [Replication Formats](https://dev.mysql.com/doc/refman/8.0/en/replication-formats.html)

**CDC Tools:**
- [Debezium Documentation](https://debezium.io/documentation/)
- [Kafka Connect](https://kafka.apache.org/documentation/#connect)
- [Maxwell's Daemon (MySQL)](https://maxwells-daemon.io/)

**Monitoring:**
- [pgMonitor](https://github.com/CrunchyData/pgmonitor)
- [Percona Monitoring and Management](https://www.percona.com/software/database-tools/percona-monitoring-and-management)

---

*This guide reflects best practices as of PostgreSQL 17 and MySQL 8.0. Always consult official documentation for your specific version.*


That's all the remaining sections! You can now copy this complete block along with all the previous parts to have your full blog post.
