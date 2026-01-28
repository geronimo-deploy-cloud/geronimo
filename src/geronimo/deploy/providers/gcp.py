"""GCP deployment provider using Pulumi."""

from typing import Optional
import pulumi
from pulumi import automation as auto
from pulumi_gcp import storage, cloudrun, artifactregistry, workflows, cloudscheduler, serviceaccount, projects

from geronimo.deploy.config import DeploymentConfig


# =============================================================================
# Artifact Storage Component (GCS)
# =============================================================================

def _create_artifact_storage(config: DeploymentConfig):
    """Create GCS bucket for artifact storage."""
    project = config.project
    
    bucket = storage.Bucket(
        f"{project}-artifacts",
        name=f"{config.artifacts.bucket_prefix}-{project}",
        location=config.region.upper(),
        versioning=storage.BucketVersioningArgs(
            enabled=config.artifacts.versioning,
        ),
        lifecycle_rules=[
            storage.BucketLifecycleRuleArgs(
                action=storage.BucketLifecycleRuleActionArgs(type="Delete"),
                condition=storage.BucketLifecycleRuleConditionArgs(
                    age=config.artifacts.retention_days,
                ),
            ),
        ] if config.artifacts.retention_days > 0 else [],
        labels={"project": project, "managed-by": "geronimo"},
    )
    
    return bucket


# =============================================================================
# Serving Infrastructure Component (Cloud Run)
# =============================================================================

def _create_serving_infra(config: DeploymentConfig):
    """Create Cloud Run serving infrastructure."""
    serving = config.serving
    project = config.project
    
    # Artifact Registry Repository
    repo = artifactregistry.Repository(
        f"{project}-repo",
        repository_id=project,
        location=config.region,
        format="DOCKER",
        labels={"project": project, "managed-by": "geronimo"},
    )
    
    # Service Account for Cloud Run
    service_account = serviceaccount.Account(
        f"{project}-run-sa",
        account_id=f"{project}-run",
        display_name=f"Geronimo Cloud Run SA for {project}",
    )
    
    # Grant storage access
    projects.IAMMember(
        f"{project}-storage-access",
        project=pulumi.Config("gcp").require("project"),
        role="roles/storage.objectViewer",
        member=service_account.email.apply(lambda e: f"serviceAccount:{e}"),
    )
    
    # Cloud Run Service
    service = cloudrun.Service(
        f"{project}-service",
        name=project,
        location=config.region,
        template=cloudrun.ServiceTemplateArgs(
            spec=cloudrun.ServiceTemplateSpecArgs(
                service_account_name=service_account.email,
                containers=[
                    cloudrun.ServiceTemplateSpecContainerArgs(
                        image=repo.name.apply(
                            lambda r: f"{config.region}-docker.pkg.dev/${{PROJECT_ID}}/{r}/{project}:latest"
                        ),
                        ports=[cloudrun.ServiceTemplateSpecContainerPortArgs(
                            container_port=serving.port,
                        )],
                        resources=cloudrun.ServiceTemplateSpecContainerResourcesArgs(
                            limits={
                                "cpu": str(serving.cpu * 1000) + "m",
                                "memory": f"{serving.memory}Mi",
                            },
                        ),
                    ),
                ],
            ),
            metadata=cloudrun.ServiceTemplateMetadataArgs(
                annotations={
                    "autoscaling.knative.dev/minScale": str(serving.min_replicas),
                    "autoscaling.knative.dev/maxScale": str(serving.max_replicas),
                },
            ),
        ),
        traffics=[cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )],
    )
    
    # Make service publicly accessible (optional)
    cloudrun.IamMember(
        f"{project}-invoker",
        service=service.name,
        location=config.region,
        role="roles/run.invoker",
        member="allUsers",
    )
    
    return {
        "repository": repo,
        "service": service,
        "service_account": service_account,
    }


# =============================================================================
# Batch Infrastructure Component (Cloud Workflows + Scheduler)
# =============================================================================

def _create_batch_infra(config: DeploymentConfig):
    """Create Cloud Workflows batch infrastructure."""
    batch = config.batch
    project = config.project
    
    # Service Account for Workflows
    workflow_sa = serviceaccount.Account(
        f"{project}-workflow-sa",
        account_id=f"{project}-workflow",
        display_name=f"Geronimo Workflow SA for {project}",
    )
    
    # Grant Cloud Run invoker role
    projects.IAMMember(
        f"{project}-workflow-run-invoker",
        project=pulumi.Config("gcp").require("project"),
        role="roles/run.invoker",
        member=workflow_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )
    
    # Cloud Workflow
    workflow = workflows.Workflow(
        f"{project}-batch-workflow",
        name=f"{project}-batch",
        region=config.region,
        service_account=workflow_sa.email,
        source_contents=f"""
main:
  steps:
    - init:
        assign:
          - project: "{project}"
    - run_batch:
        call: http.post
        args:
          url: ${{CLOUD_RUN_URL}}/batch
          auth:
            type: OIDC
        result: batch_result
    - return_result:
        return: ${{batch_result}}
""",
        labels={"project": project, "managed-by": "geronimo"},
    )
    
    # Service Account for Scheduler
    scheduler_sa = serviceaccount.Account(
        f"{project}-scheduler-sa",
        account_id=f"{project}-scheduler",
        display_name=f"Geronimo Scheduler SA for {project}",
    )
    
    # Grant workflow invoker role
    projects.IAMMember(
        f"{project}-scheduler-workflow-invoker",
        project=pulumi.Config("gcp").require("project"),
        role="roles/workflows.invoker",
        member=scheduler_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )
    
    # Cloud Scheduler Job
    schedule = cloudscheduler.Job(
        f"{project}-schedule",
        name=f"{project}-batch-schedule",
        region=config.region,
        schedule=batch.schedule,
        time_zone="UTC",
        http_target=cloudscheduler.JobHttpTargetArgs(
            uri=workflow.id.apply(
                lambda w: f"https://workflowexecutions.googleapis.com/v1/{w}/executions"
            ),
            http_method="POST",
            oauth_token=cloudscheduler.JobHttpTargetOauthTokenArgs(
                service_account_email=scheduler_sa.email,
            ),
        ),
    )
    
    return {
        "workflow": workflow,
        "schedule": schedule,
    }


# =============================================================================
# Main Deployment Functions
# =============================================================================

def _create_pulumi_program(config: DeploymentConfig, component: Optional[str]):
    """Create the Pulumi program function."""
    
    def pulumi_program():
        outputs = {}
        
        # Artifact Storage
        if component is None or component == "artifacts":
            bucket = _create_artifact_storage(config)
            outputs["artifact_bucket"] = bucket.name
            outputs["artifact_bucket_url"] = bucket.url
        
        # Serving Infrastructure
        if component is None or component == "serving":
            if config.serving:
                serving = _create_serving_infra(config)
                outputs["artifact_registry"] = serving["repository"].name
                outputs["cloud_run_url"] = serving["service"].statuses[0].url
        
        # Batch Infrastructure
        if component is None or component == "batch":
            if config.batch:
                batch = _create_batch_infra(config)
                outputs["workflow_name"] = batch["workflow"].name
                outputs["schedule_name"] = batch["schedule"].name
        
        # Export outputs
        for key, value in outputs.items():
            pulumi.export(key, value)
    
    return pulumi_program


def deploy_gcp(config: DeploymentConfig, component: Optional[str] = None) -> dict:
    """Deploy to GCP using Pulumi Automation API."""
    project_name = f"geronimo-{config.project}"
    stack_name = config.stack_name
    
    stack = auto.create_or_select_stack(
        stack_name=stack_name,
        project_name=project_name,
        program=_create_pulumi_program(config, component),
    )
    
    stack.set_config("gcp:region", auto.ConfigValue(value=config.region))
    
    print(f"Deploying {project_name} to GCP ({config.region})...")
    up_result = stack.up(on_output=print)
    
    return {
        "summary": up_result.summary,
        "outputs": {k: v.value for k, v in up_result.outputs.items()},
    }


def destroy_gcp(config: DeploymentConfig) -> dict:
    """Destroy GCP infrastructure."""
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
