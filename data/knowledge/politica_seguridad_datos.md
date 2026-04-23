# Politica de Seguridad de Datos - NovaRetail

## Clasificacion de datos
- Publico: informacion de catalogo y comunicados oficiales.
- Interno: procesos operativos y reportes no sensibles.
- Confidencial: datos de clientes, transacciones y datos personales.
- Restringido: credenciales, llaves API, secretos y datos regulados.

## Reglas para sistemas RAG
- Queda prohibido indexar secretos, llaves privadas o tokens de acceso.
- Los documentos con PII deben anonimizarse antes de ingresar al indice.
- El acceso al indice debe requerir autenticacion y registro de auditoria.
- Se debe conservar trazabilidad de fuente para toda respuesta.

## Retencion y borrado
- Logs de prompts y respuestas: retencion maxima de 90 dias.
- Vectores derivados de documentos obsoletos deben regenerarse o borrarse.
- Se debe permitir borrado selectivo por solicitud legal.

## Cumplimiento
- El sistema debe cumplir principios de minimizacion de datos.
- Auditorias trimestrales verifican sesgo, fugas y controles de acceso.
- Incidentes de seguridad deben reportarse en menos de 24 horas.
