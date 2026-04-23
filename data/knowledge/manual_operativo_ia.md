# Manual Operativo de IA - NovaRetail

## Objetivo
NovaRetail usa IA para mejorar soporte al cliente, clasificacion automatica de tickets y recomendacion de productos.
El objetivo empresarial es reducir el tiempo promedio de resolucion en un 25% sin degradar calidad.

## Politica de uso responsable
- Todo sistema de IA debe tener responsable de negocio y responsable tecnico asignados.
- Ningun modelo puede desplegarse sin evaluacion de riesgo y aprobacion de seguridad.
- Las respuestas al cliente deben pasar por filtros de seguridad y tono corporativo.

## Niveles de servicio
- SLA de latencia para respuestas asistidas por IA: p95 menor a 4 segundos.
- Disponibilidad mensual objetivo: 99.5%.
- Si la confianza del modelo cae por debajo del umbral definido, se enruta a agente humano.

## Flujo de operacion
1. Ingesta de documentos validados por el area legal y operaciones.
2. Construccion del indice vectorial.
3. Recuperacion top-k de pasajes relevantes.
4. Generacion de respuesta con citas.
5. Registro de logs para auditoria.

## Criterios de escalamiento a humano
- Solicitudes con impacto financiero mayor a 500 USD.
- Casos con lenguaje abusivo o riesgo reputacional.
- Preguntas fuera de politica o sin contexto suficiente.
