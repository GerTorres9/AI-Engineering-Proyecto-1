"""Base de datos operativa asíncrona (aiosqlite) para consulta verificada de pedidos.

Implementa los controles del Agente Especialista de Consulta del Pedido (Módulo 6):
- Consultas exclusivamente asíncronas y de solo lectura.
- Enmascaramiento de correo electrónico para cumplir SEC-004.
- Validación de entrada y salida con modelos Pydantic (`OrderLookupInput`, `OrderStatusResult`).
"""

from __future__ import annotations

from pathlib import Path
import aiosqlite

from intelligence_system.config import Settings, get_settings
from intelligence_system.schemas import OrderLookupInput, OrderStatusResult

SEED_ORDERS = [
    (
        "ORD-1001",
        "CUST-01",
        "ana.martinez@ejemplo.mx",
        "Monitor Pro 27 4K",
        "in_transit",
        "En tránsito — Centro de distribución Querétaro",
        "2026-10-20T14:30:00-06:00",
        "none",
    ),
    (
        "ORD-1002",
        "CUST-02",
        "carlos.ruiz@ejemplo.mx",
        "Laptop Studio 16GB",
        "delayed",
        "Demora logística reportada en ruta Monterrey — Revisión prioritaria",
        "2026-10-19T18:15:00-06:00",
        "none",
    ),
    (
        "ORD-1003",
        "CUST-03",
        "laura.gomez@ejemplo.mx",
        "Teclado Mecánico RGB",
        "refund_pending",
        "Devolución recibida y aprobada en almacén central",
        "2026-10-20T09:00:00-06:00",
        "approved_processing_5_to_7_business_days",
    ),
    (
        "ORD-1004",
        "CUST-04",
        "pedro.sanchez@ejemplo.mx",
        "Audífonos Inalámbricos ANC",
        "delivered",
        "Entregado en domicilio y firmado por el destinatario",
        "2026-10-15T11:45:00-06:00",
        "none",
    ),
    (
        "ORD-1005",
        "CUST-05",
        "sofia.torres@ejemplo.mx",
        "Silla Ergonómica Ejecutiva",
        "preparing",
        "Orden pagada — En preparación y empaque de almacén",
        "2026-10-21T08:20:00-06:00",
        "duplicate_charge_under_review",
    ),
]


def _mask_email(email: str) -> str:
    """Enmascara el correo electrónico para cumplir con la política de privacidad SEC-004."""
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    prefix = local[:2] if len(local) >= 2 else local[:1]
    return f"{prefix}***@{domain}"


async def init_orders_db(settings: Settings | None = None) -> int:
    """Inicializa y puebla la tabla de pedidos en SQLite de forma 100% asíncrona."""
    cfg = settings or get_settings()
    db_path = Path(cfg.orders_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                product_name TEXT NOT NULL,
                order_status TEXT NOT NULL,
                last_shipping_event TEXT NOT NULL,
                last_update TEXT NOT NULL,
                refund_status TEXT NOT NULL
            )
            """
        )
        await db.executemany(
            """
            INSERT OR REPLACE INTO orders (
                order_id, customer_id, customer_email, product_name,
                order_status, last_shipping_event, last_update, refund_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            SEED_ORDERS,
        )
        await db.commit()
        async with db.execute("SELECT COUNT(*) FROM orders") as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0


async def get_order_by_id_async(
    payload: OrderLookupInput, settings: Settings | None = None
) -> OrderStatusResult:
    """Consulta asíncrona de solo lectura validada con Pydantic."""
    cfg = settings or get_settings()
    db_path = Path(cfg.orders_db_path)
    if not db_path.exists():
        await init_orders_db(cfg)

    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT order_id, customer_id, customer_email, product_name,
                   order_status, last_shipping_event, last_update, refund_status
            FROM orders
            WHERE order_id = ?
            """,
            (payload.order_id,),
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return OrderStatusResult(
            order_found=False,
            order_id=payload.order_id,
            order_status="not_found",
            last_shipping_event="Pedido no encontrado en la base de datos autorizada.",
            last_update="",
            refund_status="none",
            source="orders_sqlite_async",
            verified=False,
        )

    return OrderStatusResult(
        order_found=True,
        order_id=str(row["order_id"]),
        customer_id=str(row["customer_id"]),
        customer_email_masked=_mask_email(str(row["customer_email"])),
        product_name=str(row["product_name"]),
        order_status=row["order_status"],
        last_shipping_event=str(row["last_shipping_event"]),
        last_update=str(row["last_update"]),
        refund_status=str(row["refund_status"]),
        source="orders_sqlite_async",
        verified=True,
    )
