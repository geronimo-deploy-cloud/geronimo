"""Tests for geronimo.batch module."""

import pytest

from geronimo.batch import BatchPipeline, Schedule, Trigger


class TestSchedule:
    """Tests for Schedule class."""

    def test_cron_schedule(self):
        """Test creating a cron schedule."""
        schedule = Schedule.cron("0 * * * *")  # Every hour
        assert schedule.cron_expression == "0 * * * *"

    def test_daily_schedule(self):
        """Test creating a daily schedule."""
        schedule = Schedule.daily(hour=6, minute=30)
        assert "6" in schedule.cron_expression
        assert "Daily" in schedule.description

    def test_weekly_schedule(self):
        """Test creating a weekly schedule."""
        schedule = Schedule.weekly(day=0, hour=12)
        assert "12" in schedule.cron_expression
        assert "Sunday" in schedule.description


class TestTrigger:
    """Tests for Trigger class."""

    def test_s3_upload_trigger(self):
        """Test creating an S3 upload trigger."""
        trigger = Trigger.s3_upload(bucket="data-bucket", prefix="input/")
        assert trigger.trigger_type.value == "s3_upload"
        assert trigger.config["bucket"] == "data-bucket"

    def test_sns_trigger(self):
        """Test creating an SNS trigger."""
        trigger = Trigger.sns_message(topic_arn="arn:aws:sns:us-east-1:123:topic")
        assert trigger.trigger_type.value == "sns_message"

    def test_manual_trigger(self):
        """Test creating a manual trigger."""
        trigger = Trigger.manual()
        assert trigger.trigger_type.value == "manual"


class TestBatchPipeline:
    """Tests for BatchPipeline class."""

    def test_pipeline_subclass(self):
        """Test creating a pipeline subclass."""
        class TestPipeline(BatchPipeline):
            name = "test-pipeline"
            schedule = Schedule.cron("0 * * * *")

            def run(self, **kwargs):
                return {"status": "completed"}

        pipeline = TestPipeline()
        assert pipeline.name == "test-pipeline"

    def test_pipeline_run(self):
        """Test running a pipeline."""
        class SimplePipeline(BatchPipeline):
            name = "simple"
            schedule = Schedule.daily(hour=6)

            def __init__(self):
                super().__init__()
                self.executed = False

            def run(self, **kwargs):
                self.executed = True
                return {"rows_processed": 100}

        pipeline = SimplePipeline()
        result = pipeline.run()
        
        assert pipeline.executed is True
        assert result["rows_processed"] == 100
