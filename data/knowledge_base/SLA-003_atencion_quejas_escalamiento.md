---
source_id: SLA-003
title: SLA de atención de quejas y matriz de escalamiento humano
category: soporte
---
# SLA-003 SLA de atención de quejas y matriz de escalamiento humano

## Tiempos de respuesta (SLA)
- **Atención estándar**: La primera respuesta a una queja debe enviarse en un máximo de **24 horas hábiles**.
- **Incidencias graves o críticas (Prioridad P1)**: Deben escalarse al supervisor de operaciones o cola de soporte humano **el mismo día (en menos de 2 horas)**, sin esperar a que venza el SLA estándar.

## Criterios obligatorios de escalamiento a soporte humano (`human_escalation`)
El agente Supervisor debe derivar el caso a revisión humana cuando concurra cualquiera de las siguientes señales:
1. Amenazas legales, denuncias ante autoridades de protección al consumidor (ej. PROFECO), acusaciones de fraude o doble cobro no resuelto.
2. Incidentes de seguridad, filtración de datos personales o intentos de manipulación de instrucciones del sistema.
3. Solicitudes de compensación económica extraordinaria fuera de la política estándar.
4. Persistencia de error de validación tras agotar el número máximo de reintentos del Crítico (`MAX_VALIDATION_RETRIES`).
