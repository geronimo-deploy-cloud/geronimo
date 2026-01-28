"""Azure deployment provider using Pulumi."""

from typing import Optional
import pulumi
from pulumi import automation as auto
from pulumi_azure_native import storage, containerregistry, app, resources, authorization

from geronimo.deploy.config import DeploymentConfig


# =============================================================================
# Artifact Storage Component (Azure Blob Storage)
# =============================================================================

def _create_artifact_storage(config: DeploymentConfig):
    """Create Azure Blob Storage for artifacts."""
    project = config.project
    safe_name = project.replace("-", "").replace("_", "")[:20]
    
    # Resource Group
    resource_group = resources.ResourceGroup(
        f"{project}-rg",
        resource_group_name=f"rg-geronimo-{project}",
        location=config.region,
        tags={"project": project, "managed-by": "geronimo"},
    )
    
    # Storage Account
    storage_account = storage.StorageAccount(
        f"{project}-storage",
        account_name=f"st{safe_name}",
        resource_group_name=resource_group.name,
        location=config.region,
        sku=storage.SkuArgs(name=storage.SkuName.STANDARD_LRS),
        kind=storage.Kind.STORAGE_V2,
        tags={"project": project, "managed-by": "geronimo"},
    )
    
    # Blob Container
    container = storage.BlobContainer(
        f"{project}-artifacts-container",
        container_name="artifacts",
        account_name=storage_account.name,
        resource_group_name=resource_group.name,
    )
    
    # Enable versioning via management policy
    storage.ManagementPolicy(
        f"{project}-lifecycle",
        account_name=storage_account.name,
        resource_group_name=resource_group.name,
        management_policy_name="default",
        policy=storage.ManagementPolicySchemaArgs(
            rules=[
                storage.ManagementPolicyRuleArgs(
                    name="retention",
                    type="Lifecycle",
                    definition=storage.ManagementPolicyDefinitionArgs(
                        actions=storage.ManagementPolicyActionArgs(
                            base_blob=storage.ManagementPolicyBaseBlobArgs(
                                delete=storage.DateAfterModificationArgs(
                                    days_after_modification_greater_than=config.artifacts.retention_days,
                                ),
                            ),
                        ),
                        filters=storage.ManagementPolicyFilterArgs(
                            blob_types=["blockBlob"],
                        ),
                    ),
                    enabled=True,
                ),
            ],
        ),
    )
    
    return {
        "resource_group": resource_group,
        "storage_account": storage_account,
        "container": container,
    }


# =============================================================================
# Serving Infrastructure Component (Azure Container Apps)
# =============================================================================

def _create_serving_infra(config: DeploymentConfig, resource_group):
    """Create Azure Container Apps serving infrastructure."""
    serving = config.serving
    project = config.project
    safe_name = project.replace("-", "").replace("_", "")[:20]
    
    # Container Registry
    registry = containerregistry.Registry(
        f"{project}-acr",
        registry_name=f"acr{safe_name}",
        resource_group_name=resource_group.name,
        location=config.region,
        sku=containerregistry.SkuArgs(name="Basic"),
        admin_user_enabled=True,
        tags={"project": project, "managed-by": "geronimo"},
    )
    
    # Container Apps Environment
    environment = app.ManagedEnvironment(
        f"{project}-env",
        environment_name=f"env-{project}",
        resource_group_name=resource_group.name,
        location=config.region,
        tags={"project": project, "managed-by": "geronimo"},
    )
    
    # Container App
    container_app = app.ContainerApp(
        f"{project}-app",
        container_app_name=project,
        resource_group_name=resource_group.name,
        location=config.region,
        managed_environment_id=environment.id,
        configuration=app.ConfigurationArgs(
            ingress=app.IngressArgs(
                external=True,
                target_port=serving.port,
                traffic=[app.TrafficWeightArgs(
                    latest_revision=True,
                    weight=100,
                )],
            ),
            registries=[app.RegistryCredentialsArgs(
                server=registry.login_server,
                username=registry.name,
                password_secret_ref="acr-password",
            )],
            secrets=[app.SecretArgs(
                name="acr-password",
                value=registry.name.apply(lambda n: "placeholder"),  # Set via external secret
            )],
        ),
        template=app.TemplateArgs(
            containers=[app.ContainerArgs(
                name=project,
                image=registry.login_server.apply(lambda s: f"{s}/{project}:latest"),
                resources=app.ContainerResourcesArgs(
                    cpu=serving.cpu,
                    memory=f"{serving.memory / 1024:.1f}Gi",
                ),
            )],
            scale=app.ScaleArgs(
                min_replicas=serving.min_replicas,
                max_replicas=serving.max_replicas,
            ),
        ),
        tags={"project": project, "managed-by": "geronimo"},
    )
    
    return {
        "registry": registry,
        "environment": environment,
        "container_app": container_app,
    }


# =============================================================================
# Batch Infrastructure Component (Azure Container Instances + Logic Apps)
# =============================================================================

def _create_batch_infra(config: DeploymentConfig, resource_group):
    """Create Azure batch infrastructure with Logic Apps scheduling."""
    batch = config.batch
    project = config.project
    
    # For batch, we use Azure Container Instances triggered by Logic Apps
    # This is a simplified implementation - production would use Azure Functions or Durable Functions
    
    # Note: Logic Apps require ARM template deployment in Pulumi
    # For now, we create a placeholder that can be extended
    
    pulumi.log.info(
        f"Azure batch infrastructure for {project}: "
        "Use Azure Logic Apps or Azure Functions for scheduling. "
        "See Azure documentation for timer-triggered batch jobs."
    )
    
    return {
        "message": "Azure batch requires Logic Apps or Functions - configure manually",
    }


# =============================================================================
# Main Deployment Functions
# =============================================================================

def _create_pulumi_program(config: DeploymentConfig, component: Optional[str]):
    """Create the Pulumi program function."""
    
    def pulumi_program():
        outputs = {}
        resource_group = None
        
        # Artifact Storage (always creates resource group)
        if component is None or component == "artifacts":
            storage_infra = _create_artifact_storage(config)
            resource_group = storage_infra["resource_group"]
            outputs["resource_group"] = resource_group.name
            outputs["storage_account"] = storage_infra["storage_account"].name
            outputs["artifact_container"] = storage_infra["container"].name
        
        # Serving Infrastructure
        if component is None or component == "serving":
            if config.serving:
                if resource_group is None:
                    resource_group = resources.ResourceGroup(
                        f"{config.project}-rg",
                        resource_group_name=f"rg-geronimo-{config.project}",
                        location=config.region,
                    )
                serving = _create_serving_infra(config, resource_group)
                outputs["acr_login_server"] = serving["registry"].login_server
                outputs["container_app_url"] = serving["container_app"].configuration.apply(
                    lambda c: c.ingress.fqdn if c and c.ingress else "pending"
                )
        
        # Batch Infrastructure
        if component is None or component == "batch":
            if config.batch:
                if resource_group is None:
                    resource_group = resources.ResourceGroup(
                        f"{config.project}-rg",
                        resource_group_name=f"rg-geronimo-{config.project}",
                        location=config.region,
                    )
                _create_batch_infra(config, resource_group)
        
        # Export outputs
        for key, value in outputs.items():
            pulumi.export(key, value)
    
    return pulumi_program


def deploy_azure(config: DeploymentConfig, component: Optional[str] = None) -> dict:
    """Deploy to Azure using Pulumi Automation API."""
    project_name = f"geronimo-{config.project}"
    stack_name = config.stack_name
    
    stack = auto.create_or_select_stack(
        stack_name=stack_name,
        project_name=project_name,
        program=_create_pulumi_program(config, component),
    )
    
    stack.set_config("azure-native:location", auto.ConfigValue(value=config.region))
    
    print(f"Deploying {project_name} to Azure ({config.region})...")
    up_result = stack.up(on_output=print)
    
    return {
        "summary": up_result.summary,
        "outputs": {k: v.value for k, v in up_result.outputs.items()},
    }


def destroy_azure(config: DeploymentConfig) -> dict:
    """Destroy Azure infrastructure."""
    project_name = f"geronimo-{config.project}"
    stack_name = config.stack_name
    
    stack = auto.select_stack(
        stack_name=stack_name,
        project_name=project_name,
        program=lambda: None,
    )
    
    print(f"Destroying {project_name}...")
    destroy_result = stack.destroy(on_output=print)
    
    return {"summary": destroy_result.summary}
