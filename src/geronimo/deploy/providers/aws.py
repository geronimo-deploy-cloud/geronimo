"""AWS deployment provider using Pulumi."""

from typing import Optional
import pulumi
from pulumi import automation as auto
from pulumi_aws import s3, ecs, ecr, iam, lb, cloudwatch, sfn, scheduler

from geronimo.deploy.config import DeploymentConfig


# =============================================================================
# Artifact Storage Component
# =============================================================================

def _create_artifact_storage(config: DeploymentConfig):
    """Create S3 bucket for artifact storage."""
    bucket_name = f"{config.artifacts.bucket_prefix}-{config.project}"
    
    bucket = s3.BucketV2(
        f"{config.project}-artifacts",
        bucket=bucket_name,
        tags={"Project": config.project, "ManagedBy": "geronimo"},
    )
    
    s3.BucketVersioningV2(
        f"{config.project}-artifacts-versioning",
        bucket=bucket.id,
        versioning_configuration=s3.BucketVersioningV2VersioningConfigurationArgs(
            status="Enabled" if config.artifacts.versioning else "Disabled",
        ),
    )
    
    if config.artifacts.retention_days > 0:
        s3.BucketLifecycleConfigurationV2(
            f"{config.project}-artifacts-lifecycle",
            bucket=bucket.id,
            rules=[
                s3.BucketLifecycleConfigurationV2RuleArgs(
                    id="retention",
                    status="Enabled",
                    expiration=s3.BucketLifecycleConfigurationV2RuleExpirationArgs(
                        days=config.artifacts.retention_days,
                    ),
                    noncurrent_version_expiration=s3.BucketLifecycleConfigurationV2RuleNoncurrentVersionExpirationArgs(
                        noncurrent_days=config.artifacts.retention_days,
                    ),
                ),
            ],
        )
    
    return bucket


# =============================================================================
# Serving Infrastructure Component (ECS Fargate)
# =============================================================================

def _create_serving_infra(config: DeploymentConfig):
    """Create ECS Fargate serving infrastructure."""
    serving = config.serving
    project = config.project
    
    # ECR Repository
    repo = ecr.Repository(
        f"{project}-repo",
        name=project,
        image_tag_mutability="MUTABLE",
        image_scanning_configuration=ecr.RepositoryImageScanningConfigurationArgs(
            scan_on_push=True,
        ),
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    # ECS Cluster
    cluster = ecs.Cluster(
        f"{project}-cluster",
        name=f"{project}-cluster",
        settings=[ecs.ClusterSettingArgs(name="containerInsights", value="enabled")],
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    # Task Execution Role
    exec_role = iam.Role(
        f"{project}-exec-role",
        name=f"{project}-ecs-exec-role",
        assume_role_policy="""{
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Effect": "Allow"
            }]
        }""",
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    iam.RolePolicyAttachment(
        f"{project}-exec-policy",
        role=exec_role.name,
        policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )
    
    # Task Role (for application code)
    task_role = iam.Role(
        f"{project}-task-role",
        name=f"{project}-ecs-task-role",
        assume_role_policy="""{
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Effect": "Allow"
            }]
        }""",
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    # Allow task to read from S3 artifacts bucket
    iam.RolePolicy(
        f"{project}-task-s3-policy",
        role=task_role.name,
        policy=pulumi.Output.all(config.artifacts.bucket_prefix, project).apply(
            lambda args: f"""{{
                "Version": "2012-10-17",
                "Statement": [{{
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        "arn:aws:s3:::{args[0]}-{args[1]}",
                        "arn:aws:s3:::{args[0]}-{args[1]}/*"
                    ]
                }}]
            }}"""
        ),
    )
    
    # CloudWatch Log Group
    log_group = cloudwatch.LogGroup(
        f"{project}-logs",
        name=f"/ecs/{project}",
        retention_in_days=30,
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    # Task Definition
    task_def = ecs.TaskDefinition(
        f"{project}-task",
        family=project,
        cpu=str(serving.cpu * 1024),  # Convert vCPU to CPU units
        memory=str(serving.memory),
        network_mode="awsvpc",
        requires_compatibilities=["FARGATE"],
        execution_role_arn=exec_role.arn,
        task_role_arn=task_role.arn,
        container_definitions=pulumi.Output.all(repo.repository_url, log_group.name).apply(
            lambda args: f"""[{{
                "name": "{project}",
                "image": "{args[0]}:latest",
                "portMappings": [{{"containerPort": {serving.port}, "protocol": "tcp"}}],
                "logConfiguration": {{
                    "logDriver": "awslogs",
                    "options": {{
                        "awslogs-group": "{args[1]}",
                        "awslogs-region": "{config.region}",
                        "awslogs-stream-prefix": "ecs"
                    }}
                }},
                "essential": true
            }}]"""
        ),
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    return {
        "ecr_repo": repo,
        "cluster": cluster,
        "task_definition": task_def,
        "log_group": log_group,
    }


# =============================================================================
# Batch Infrastructure Component (Step Functions)
# =============================================================================

def _create_batch_infra(config: DeploymentConfig):
    """Create Step Functions batch infrastructure."""
    batch = config.batch
    project = config.project
    
    # IAM Role for Step Functions
    sfn_role = iam.Role(
        f"{project}-sfn-role",
        name=f"{project}-sfn-role",
        assume_role_policy="""{
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "states.amazonaws.com"},
                "Effect": "Allow"
            }]
        }""",
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    # Allow Step Functions to run ECS tasks
    iam.RolePolicy(
        f"{project}-sfn-ecs-policy",
        role=sfn_role.name,
        policy="""{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecs:RunTask",
                        "ecs:StopTask",
                        "ecs:DescribeTasks",
                        "iam:PassRole",
                        "logs:CreateLogDelivery",
                        "logs:GetLogDelivery",
                        "logs:UpdateLogDelivery",
                        "logs:DeleteLogDelivery",
                        "logs:ListLogDeliveries",
                        "logs:PutResourcePolicy",
                        "logs:DescribeResourcePolicies",
                        "logs:DescribeLogGroups"
                    ],
                    "Resource": "*"
                }
            ]
        }""",
    )
    
    # CloudWatch Log Group for Step Functions
    sfn_log_group = cloudwatch.LogGroup(
        f"{project}-sfn-logs",
        name=f"/aws/vendedlogs/states/{project}-batch",
        retention_in_days=30,
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    # State Machine Definition (placeholder - runs a simple pass state)
    state_machine = sfn.StateMachine(
        f"{project}-batch-pipeline",
        name=f"{project}-batch-pipeline",
        role_arn=sfn_role.arn,
        definition=f"""{{
            "Comment": "Geronimo batch pipeline for {project}",
            "StartAt": "RunBatchJob",
            "States": {{
                "RunBatchJob": {{
                    "Type": "Task",
                    "Resource": "arn:aws:states:::ecs:runTask.sync",
                    "Parameters": {{
                        "LaunchType": "FARGATE",
                        "Cluster.$": "$.cluster_arn",
                        "TaskDefinition.$": "$.task_definition_arn",
                        "NetworkConfiguration": {{
                            "AwsvpcConfiguration": {{
                                "Subnets.$": "$.subnets",
                                "AssignPublicIp": "ENABLED"
                            }}
                        }}
                    }},
                    "End": true
                }}
            }}
        }}""",
        logging_configuration=sfn.StateMachineLoggingConfigurationArgs(
            log_destination=sfn_log_group.arn.apply(lambda arn: f"{arn}:*"),
            include_execution_data=True,
            level="ALL",
        ),
        tags={"Project": project, "ManagedBy": "geronimo"},
        type="STANDARD",
    )
    
    # EventBridge Scheduler Role
    scheduler_role = iam.Role(
        f"{project}-scheduler-role",
        name=f"{project}-scheduler-role",
        assume_role_policy="""{
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "scheduler.amazonaws.com"},
                "Effect": "Allow"
            }]
        }""",
        tags={"Project": project, "ManagedBy": "geronimo"},
    )
    
    iam.RolePolicy(
        f"{project}-scheduler-sfn-policy",
        role=scheduler_role.name,
        policy=state_machine.arn.apply(lambda arn: f"""{{
            "Version": "2012-10-17",
            "Statement": [{{
                "Effect": "Allow",
                "Action": "states:StartExecution",
                "Resource": "{arn}"
            }}]
        }}"""),
    )
    
    # EventBridge Schedule
    schedule = scheduler.Schedule(
        f"{project}-schedule",
        name=f"{project}-batch-schedule",
        schedule_expression=f"cron({batch.schedule})",
        flexible_time_window=scheduler.ScheduleFlexibleTimeWindowArgs(
            mode="OFF",
        ),
        target=scheduler.ScheduleTargetArgs(
            arn=state_machine.arn,
            role_arn=scheduler_role.arn,
            input="{}",
        ),
    )
    
    return {
        "state_machine": state_machine,
        "schedule": schedule,
        "sfn_log_group": sfn_log_group,
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
            outputs["artifact_bucket"] = bucket.bucket
            outputs["artifact_bucket_arn"] = bucket.arn
        
        # Serving Infrastructure
        if component is None or component == "serving":
            if config.serving:
                serving = _create_serving_infra(config)
                outputs["ecr_repo_url"] = serving["ecr_repo"].repository_url
                outputs["ecs_cluster_arn"] = serving["cluster"].arn
                outputs["task_definition_arn"] = serving["task_definition"].arn
        
        # Batch Infrastructure
        if component is None or component == "batch":
            if config.batch:
                batch = _create_batch_infra(config)
                outputs["state_machine_arn"] = batch["state_machine"].arn
                outputs["schedule_arn"] = batch["schedule"].arn
        
        # Export outputs
        for key, value in outputs.items():
            pulumi.export(key, value)
    
    return pulumi_program


def deploy_aws(config: DeploymentConfig, component: Optional[str] = None) -> dict:
    """Deploy to AWS using Pulumi Automation API."""
    project_name = f"geronimo-{config.project}"
    stack_name = config.stack_name
    
    stack = auto.create_or_select_stack(
        stack_name=stack_name,
        project_name=project_name,
        program=_create_pulumi_program(config, component),
    )
    
    stack.set_config("aws:region", auto.ConfigValue(value=config.region))
    
    print(f"Deploying {project_name} to AWS ({config.region})...")
    up_result = stack.up(on_output=print)
    
    return {
        "summary": up_result.summary,
        "outputs": {k: v.value for k, v in up_result.outputs.items()},
    }


def destroy_aws(config: DeploymentConfig) -> dict:
    """Destroy AWS infrastructure."""
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
