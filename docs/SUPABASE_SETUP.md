# Supabase PostgreSQL setup

Supabase provides a managed PostgreSQL database, so RazorRisk can use it as the shared application data plane instead of storing transactions in a local SQLite file. Supabase documents its database as full PostgreSQL and provides connection pooling through Supavisor. For a persistent API/worker deployment, use the **session-mode** pooler connection (or a direct connection when the deployment/network supports it).

## 1. Create the project

Create a Supabase project and open **Connect → Connection string**.

Use the session-mode connection for long-lived FastAPI/worker containers. Do not put the database password in source control.

## 2. Configure RazorRisk

Set:

```env
DATABASE_URL=postgresql://postgres.<POOLER_TENANT>:<PASSWORD>@aws-<REGION>.pooler.supabase.com:5432/postgres
```

The exact hostname, tenant and password must come from the Supabase dashboard.

## 3. Initialize the schema

From the RazorRisk repository:

```bash
python -c "from db.database import init_db; init_db()"
```

This applies `db/schema.sql` to PostgreSQL.

## 4. Seed the dataset

```bash
python -m data.generate_synthetic_data
```

The generator writes the synthetic fraud-ring dataset into PostgreSQL. There is no requirement to copy `razor_risk.db` into the deployment.

## 5. Start API + workers

Configure the same `DATABASE_URL` and `REDIS_URL` for every API replica and investigation worker. Redis remains the distributed rate-limit/job layer; PostgreSQL is the shared durable application-data layer.

## 6. Connection mode note

Supabase's transaction-mode pooler is intended for short-lived/serverless traffic. RazorRisk's FastAPI process and investigation workers are long-lived services, so session mode is the safer default for this architecture. If a deployment uses transaction pooling, validate all connection-library behavior against the provider's pooling constraints.
