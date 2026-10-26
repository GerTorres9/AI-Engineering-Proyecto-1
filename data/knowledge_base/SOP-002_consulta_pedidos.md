---
source_id: SOP-002
title: Consulta y estados oficiales de pedidos
category: pedidos
---
# SOP-002 Consulta y estados oficiales de pedidos

Para revisar el estado operativo de un pedido, el especialista de pedidos debe consultar la base de datos autorizada utilizando el número de pedido (`order_id`, ej. `ORD-1001`) y verificar la relación con el cliente antes de divulgar detalles.

## Estados permitidos en el sistema
- **Preparación (`preparing`)**: El pedido fue pagado y se encuentra en almacén para empaque.
- **En tránsito / Enviado (`in_transit`)**: El paquete fue entregado al transportista y cuenta con guía de rastreo activa.
- **Retrasado en ruta (`delayed`)**: El transportista reportó una demora logística; se debe informar el último evento confirmado y evaluar si aplica escalamiento según el SLA.
- **Entregado (`delivered`)**: El paquete fue recibido en el domicilio registrado.
- **En proceso de reembolso (`refund_pending` / `refunded`)**: El pedido tiene una devolución o reembolso asociado.

## Regla estricta contra alucinaciones operativas
No se debe inferir ni prometer una fecha exacta de entrega ni autorizar compensaciones si el sistema operativo de pedidos no las proporciona explícitamente. Si un pedido no existe en la base de datos (`not_found`), se debe declarar de forma transparente y pedir al cliente que verifique el identificador.
