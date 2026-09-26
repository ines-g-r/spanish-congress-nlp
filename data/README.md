# Processed Data

This directory contains the cleaned corpus described in the [main README](../README.md#corpus): interventions from the [Spanish Congress of Deputies portal](https://www.congreso.es/es/), with the speech text extracted directly from the session PDFs and cleaned up, enriched with their associated metadata.

## Structure

Files are organised in a folder per legislative term (XI–XV), and within each, a folder per Congress body (e.g. *Comisión de Justicia*, *Pleno*). Each JSON file is named after its source PDF identifier (`IDPDF`) — for example, *Intervenciones_DSCD-11-CO-2.json*, where "11" identifies the legislative term and "CO" the type of Congress body — and contains the interventions extracted from that session.

## Fields

Each JSON file is a list of interventions with the following fields:

| Field | Description |
| :--- | :--- |
| `ID` | Intervention identifier: PDF identifier + page number + speaker's surname(s) and first name + processed intervention number (e.g. *DSCD-12-CO-913CiuróiBuldóLourdes6021*) |
| `ORADOR` | Speaker's surname(s) and first name |
| `TEXTO` | Text assigned to the intervention (default: *No disponible*) |
| `CARGOORADOR` | Speaker's role (e.g. *Candidato a la presidencia*, *Presidenta de la mesa*, *Ministro de justicia*; default: *No disponible*) |
| `PARTIDO` | Speaker's political group (default: *No disponible*) |
| `LEGISLATURA` | Legislative term ["XI", "XII", "XIII", "XIV", "XV"] |
| `OBJETOINICIATIVA` | Specific objective (e.g. *Solicitud de comparecencia*, *Proposición de Ley*) |
| `TIPOINICIATIVA` | Parliamentary action (e.g. *Moción de censura*, *Informe del Tribunal de Cuentas*) |
| `ORGANO` | Parliamentary body (e.g. *Pleno*, *Comisión de Igualdad*) |
| `FASE` | Stage of the initiative (e.g. *Debate*, *Votación*, *Celebración*; default: *No disponible*) |
| `SESION` | Date of the intervention, "DD/MM/AAAA" (e.g. *21/12/2016*) |
| `INICIOINTERVENCION` | Start time of the intervention, "HH:MM" (default: *No disponible*) |
| `FININTERVENCION` | End time of the intervention, "HH:MM" (default: *No disponible*) |
| `IDPDF` | PDF document identifier (e.g. *DSCD-12-CO-91*) |
| `PAGPDF` | Page of the PDF document where the intervention begins |
| `PARTIDOIMPUTADO` | Boolean: whether the party was inferred (*true*) or comes from the original data (*false*) |
| `GENEROIMPUTADO` | Speaker's gender ["M", "F"] |