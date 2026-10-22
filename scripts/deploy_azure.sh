#!/usr/bin/env bash
set -euo pipefail

# Ejecutar desde la raíz del repositorio, con az login realizado previamente.
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az group create --name rg-wine-ml-demo --location westeurope
az containerapp env create \
  --name env-wine-ml-demo \
  --resource-group rg-wine-ml-demo \
  --location westeurope
az containerapp up \
  --name wine-quality-api \
  --resource-group rg-wine-ml-demo \
  --environment env-wine-ml-demo \
  --location westeurope \
  --source . \
  --ingress external \
  --target-port 8000
az containerapp update \
  --name wine-quality-api \
  --resource-group rg-wine-ml-demo \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 0 \
  --max-replicas 2
az containerapp show \
  --name wine-quality-api \
  --resource-group rg-wine-ml-demo \
  --query properties.configuration.ingress.fqdn \
  --output tsv
