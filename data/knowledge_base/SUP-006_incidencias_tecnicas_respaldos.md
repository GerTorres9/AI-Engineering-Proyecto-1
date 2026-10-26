---
source_id: SUP-006
title: Atención de incidencias técnicas y política de respaldos operativos
category: soporte
---
# SUP-006 Atención de incidencias técnicas y política de respaldos operativos

## Diagnóstico de productos y servicios técnicos
- Para atender una falla técnica de un producto adquirido, se debe registrar el **modelo del producto**, una **descripción clara del problema** y los **pasos para reproducirlo**.
- No se deben solicitar contraseñas ni datos de pago durante el soporte técnico (`SEC-004`).
- El diagnóstico inicial debe comenzar dentro de **1 día hábil** (24 horas hábiles). Si existe riesgo para varias personas o una interrupción crítica del servicio, se escala inmediatamente al equipo responsable (`SLA-003`).

## Continuidad operativa y política de respaldos
- Los sistemas de estado de conversaciones (Checkpointer) y bases operativas mantienen persistencia transaccional para no perder el contexto ante reinicios del servidor.
- Los respaldos incrementales del sistema operativo se ejecutan cada **6 horas**, mientras que los respaldos completos se realizan de forma **diaria a las 02:00 UTC** con retención de 30 días.
