"""Schedule and Trigger classes for batch pipelines."""

from enum import Enum
from typing import Optional


class Schedule:
    """Cron-based schedule for batch pipelines.

    Example:
        ```python
        from geronimo.batch import Schedule

        # Daily at 6 AM
        daily = Schedule.cron("0 6 * * *")

        # Every hour
        hourly = Schedule.cron("0 * * * *")

        # Weekly on Sunday
        weekly = Schedule.weekly(day=0, hour=0)
        ```
    """

    def __init__(self, cron_expression: str, description: Optional[str] = None):
        """Initialize schedule.

        Args:
            cron_expression: Standard cron expression (min hour day month weekday).
            description: Optional human-readable description.
        """
        self.cron_expression = cron_expression
        self.description = description

    @classmethod
    def cron(cls, expression: str) -> "Schedule":
        """Create from cron expression.

        Args:
            expression: Cron expression.

        Returns:
            Schedule instance.
        """
        return cls(cron_expression=expression)

    @classmethod
    def daily(cls, hour: int = 0, minute: int = 0) -> "Schedule":
        """Create daily schedule.

        Args:
            hour: Hour (0-23).
            minute: Minute (0-59).

        Returns:
            Schedule instance.
        """
        return cls(
            cron_expression=f"{minute} {hour} * * *",
            description=f"Daily at {hour:02d}:{minute:02d}",
        )

    @classmethod
    def weekly(cls, day: int = 0, hour: int = 0) -> "Schedule":
        """Create weekly schedule.

        Args:
            day: Day of week (0=Sunday, 6=Saturday).
            hour: Hour (0-23).

        Returns:
            Schedule instance.
        """
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return cls(
            cron_expression=f"0 {hour} * * {day}",
            description=f"Weekly on {days[day]} at {hour:02d}:00",
        )

    def __repr__(self) -> str:
        return f"Schedule({self.cron_expression})"


class TriggerType(str, Enum):
    """Types of event triggers."""

    S3_UPLOAD = "s3_upload"
    SNS_MESSAGE = "sns_message"
    MANUAL = "manual"


class Trigger:
    """Event-based trigger for batch pipelines.

    Example:
        ```python
        from geronimo.batch import Trigger

        # Trigger on S3 upload
        s3_trigger = Trigger.s3_upload(bucket="data-bucket", prefix="input/")

        # Manual trigger only
        manual = Trigger.manual()
        ```
    """

    def __init__(
        self,
        trigger_type: TriggerType,
        config: Optional[dict] = None,
    ):
        """Initialize trigger.

        Args:
            trigger_type: Type of trigger.
            config: Trigger-specific configuration.
        """
        self.trigger_type = trigger_type
        self.config = config or {}

    @classmethod
    def s3_upload(cls, bucket: str, prefix: Optional[str] = None) -> "Trigger":
        """Trigger on S3 object upload.

        Args:
            bucket: S3 bucket name.
            prefix: Optional key prefix filter.

        Returns:
            Trigger instance.
        """
        return cls(
            trigger_type=TriggerType.S3_UPLOAD,
            config={"bucket": bucket, "prefix": prefix},
        )

    @classmethod
    def sns_message(cls, topic_arn: str) -> "Trigger":
        """Trigger on SNS message.

        Args:
            topic_arn: SNS topic ARN.

        Returns:
            Trigger instance.
        """
        return cls(
            trigger_type=TriggerType.SNS_MESSAGE,
            config={"topic_arn": topic_arn},
        )

    @classmethod
    def manual(cls) -> "Trigger":
        """Manual trigger only.

        Returns:
            Trigger instance.
        """
        return cls(trigger_type=TriggerType.MANUAL)

    def __repr__(self) -> str:
        return f"Trigger({self.trigger_type.value})"
