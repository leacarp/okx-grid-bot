# OKX Grid Trading Bot

Bot de grid trading automático para OKX, desarrollado con Python 3.11+ y ccxt.

**Capital inicial:** $10 USDT en BTC/USDT  
**Estado:** Fase 1 — Solo lectura (DRY_RUN activo)

---

## Requisitos

- Python 3.11+
- Cuenta OKX con API Key configurada

## Instalación

```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus credenciales
```

## Uso

```bash
# Modo scan: lee precio y balance (sin órdenes)
python main.py --mode=scan
```

## Seguridad

- `DRY_RUN=true` por defecto. El bot NO ejecuta órdenes reales hasta configurar `DRY_RUN=false` explícitamente.
- Nunca commitear `.env`.
- API Key solo con permisos de "Trade", sin permisos de retiro.

## Estructura

```
src/
├── connectors/okx_client.py   # Wrapper ccxt con retry y DRY_RUN
├── core/price_reader.py       # Lectura de precio actual
└── utils/
    ├── logger.py              # Logging JSON con structlog
    └── config.py              # Carga y validación de config.yaml
```

## Fases

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1 | ✅ Completa | Lectura de precio y balance |
| 2 | Pendiente | Grid en DRY_RUN |
| 3 | Pendiente | Ejecución real |
