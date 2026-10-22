# Wine Quality: ML → FastAPI → Tests → Docker → Azure Container Apps

Práctica de despliegue end-to-end de Machine Learning. Dataset de **Wine Quality (vino tinto)** de UCI (1599 filas, 11 features). La etiqueta positiva es `quality >= 6`, derivada de la valoración sensorial original. Se entrena un `CatBoostClassifier` con Polars, se guarda el modelo como `artifacts/model.cbm`, y se publica una API REST con FastAPI + Pydantic. Incluye Pytest, Docker, smoke tests, un pipeline opcional de CI/CD y despliegue a Azure Container Apps.

Fuente: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality). Dataset de Cortez et al. (2009); licencia y condiciones originales en UCI.

## 0. Prerrequisitos

- Python **3.11**, Git y Docker Desktop (con WSL2 si utilizas Windows).
- Para Azure: suscripción habilitada, [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) y permisos para crear recursos.
- Para CI/CD: repositorio propio de GitHub; el CD está **desactivado por defecto** hasta que se habilite la autenticación.

Todos los comandos siguientes usan Bash/WSL (en PowerShell cambia `source .venv/bin/activate` por `.\\.venv\\Scripts\\Activate.ps1`). Ejecutar desde la carpeta raíz del proyecto.

## 1. Entrenar el modelo (aprox. 15-30 minutos con instalación)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m scripts.train
```

La primera ejecución descarga `data/winequality-red.csv` desde UCI, lee datos con `pl.scan_csv` (Lazy API), sustituye los espacios de los nombres de columnas por guiones bajos, crea el objetivo binario y entrena CatBoost. Genera:

- `artifacts/model.cbm`: fichero del modelo.
- `artifacts/metadata.json`: versión (hash del modelo), features, métricas, fecha y filas.

**Ojo**: el dataset original es muy pequeño. Polars Lazy ilustra un patrón eficiente, pero **no acredita Big Data distribuido**. La evaluación es didáctica; la calidad predicha no debe usarse para decisiones reales.

## 2. Ejecutar FastAPI local

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Navega a <http://127.0.0.1:8000/docs> (Swagger). Prueba `/health`, `/model-info` y `/predict` mediante Swagger con el JSON de `example-request.json`. En otra terminal, con entorno activado:

```bash
python -m scripts.smoke_test --base-url http://127.0.0.1:8000
```

- `POST /predict`: devuelve `quality_good`, `probability_good`, `model_version`.
- `GET /health`: liveness.
- `GET /model-info`: metadatos del modelo activo.
- Pydantic rechaza datos incorrectos con **HTTP 422**.

## 3. Ejecutar tests

```bash
python -m pytest -q
```

`tests/test_units.py`: validación de datos, orden de features y predicción con modelo simulado (Mock). `tests/test_api.py`: TestClient, ciclo de vida de aplicación, health, model-info, predicción con modelo real y errores HTTP 422. La prueba de integración **requiere entrenar antes el modelo**.

## 4. Docker local

Ejecuta después de entrenar; los artefactos se integran en la imagen y **no** se entrena dentro del contenedor de inferencia:

```bash
docker build -t wine-ml-api:1.0 .
docker run --rm -p 8000:8000 --name wine-api wine-ml-api:1.0
```

Abre <http://localhost:8000/docs> o ejecuta el smoke test contra el contenedor.

## 5. Desplegar en Azure Container Apps (manual)

Abre **otra terminal** desde la raíz del repositorio, con Azure CLI instalado:

```bash
az login
az account show --query '{name:name,id:id}' --output json
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az group create -n rg-wine-ml-demo -l westeurope
az containerapp env create -n env-wine-ml-demo -g rg-wine-ml-demo -l westeurope
az containerapp up \
  --name wine-quality-api \
  --resource-group rg-wine-ml-demo \
  --environment env-wine-ml-demo \
  --location westeurope \
  --source . \
  --ingress external \
  --target-port 8000
```

`az containerapp up --source .` crea/usa un registro para construir y publicar la imagen, y actualiza el servicio. Requiere que hayas ejecutado `python -m scripts.train` previamente (Dockerfile copia `artifacts/`).

```bash
az containerapp update -n wine-quality-api -g rg-wine-ml-demo \
  --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 2
az containerapp show -n wine-quality-api -g rg-wine-ml-demo \
  --query properties.configuration.ingress.fqdn -o tsv
```

Toma el FQDN devuelto, añade `https://` y `/docs`, y ejecuta:

```bash
python -m scripts.smoke_test --base-url https://TU_FQDN
```

Comprobar logs:

```bash
az containerapp logs show -n wine-quality-api -g rg-wine-ml-demo --type console --follow
```

El escalado a cero reduce costes de cómputo en reposo, pero implica **cold starts**. La suscripción puede incurrir en costes de Container Registry, Log Analytics u otros recursos incluso si la app tiene poca actividad. **Al terminar, borra el grupo de recursos**:

```bash
az group delete -n rg-wine-ml-demo --yes --no-wait
```

## 6. GitHub Actions: CI y CD opcional

El archivo `.github/workflows/ci-cd.yml` incluye dos trabajos:

- **test**: en push y pull_request a `main`, instala dependencias, descarga dataset, entrena modelo, lanza Pytest y construye Docker.
- **deploy**: depende de `test`, solo se ejecuta en `main` si la variable del repositorio `ENABLE_AZURE_DEPLOY` vale `true`. Sin esta variable, puedes practicar **CI sin configurar Azure**.

Publica el repositorio en GitHub:

```bash
git init
git add .
git commit -m "Initial Wine ML API"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

Para activar CD seguro mediante OIDC (opcional, probablemente requiera otra sesión si es tu primera vez):

1. En Microsoft Entra, registra una aplicación o configura una identidad administrada asignada por el usuario.
2. En Azure, asígnale permisos de despliegue **acotados al grupo de recursos de la demo**, no a toda la suscripción (por simplicidad puedes usar Contributor en ese grupo de recursos, siguiendo las políticas de tu cuenta).
3. En la identidad, configura una **federated credential** para el repo `TU_USUARIO/TU_REPO`, entidad `Branch`, rama `main`, audiencia `api://AzureADTokenExchange`.
4. GitHub → Settings → Secrets and variables → Actions → Secrets: agrega `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`.
5. GitHub → Settings → Secrets and variables → Actions → Variables: crea `ENABLE_AZURE_DEPLOY=true`.
6. Un push a `main` ejecutará tests y build **antes** de desplegar. No se almacenan contraseñas de Azure en GitHub. El CD asume que la app y su entorno se crearon en el paso 5.

[Documentación oficial OIDC](https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure-openid-connect) · [Azure Container Apps CLI](https://learn.microsoft.com/en-us/azure/container-apps/containerapp-up).

## 7. Checklist de resultados de aprendizaje

- [ ] Descargué, limpié y transformé el dataset con Polars Lazy.
- [ ] Entrené CatBoost y serialicé su modelo y métricas.
- [ ] Publiqué una API FastAPI con validación Pydantic y Swagger.
- [ ] Ejecuté unit tests y tests de integración con pytest.
- [ ] Construí y ejecuté una imagen Docker local.
- [ ] Desplegué la imagen en Azure Container Apps.
- [ ] Consulté logs y probé la API pública con smoke test.
- [ ] Configuré CI con GitHub Actions; CD opcional si dispongo de permisos.
- [ ] Borré los recursos cloud de prueba para controlar costes.

> No publiques contraseñas ni credenciales en Git; el modelo y el dataset se generan durante entrenamiento. El código es una prueba didáctica, no un stack de seguridad listo para producción.
