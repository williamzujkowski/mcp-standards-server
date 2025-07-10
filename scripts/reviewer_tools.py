#!/usr/bin/env python3
"""
Reviewer Tools for Standards Review Process

Provides utilities for managing the review process, including
reviewer assignment, progress tracking, and notification management.
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from github import Github

logger = logging.getLogger(__name__)


@dataclass
class Reviewer:
    """Information about a reviewer."""

    username: str
    name: str
    domains: list[str]
    role: str  # maintainer, domain_expert, editorial, community
    max_concurrent: int = 3
    current_assignments: int = 0
    availability: bool = True
    contact_info: str | None = None


@dataclass
class ReviewAssignment:
    """A review assignment."""

    standard_name: str
    reviewer: str
    stage: str  # technical, editorial, community, final
    assigned_date: datetime
    due_date: datetime
    status: str  # assigned, in_progress, completed, overdue
    feedback_url: str | None = None


class ReviewerManager:
    """Manages reviewer assignments and tracking."""

    def __init__(self, config_file: str = "reviewer_config.yaml"):
        """Initialize with reviewer configuration."""
        self.config_file = Path(config_file)
        self.reviewers: dict[str, Reviewer] = {}
        self.assignments: list[ReviewAssignment] = []
        self.github = None

        self._load_config()
        self._load_assignments()

    def _load_config(self):
        """Load reviewer configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                config = yaml.safe_load(f)

            # Load reviewers
            for reviewer_data in config.get("reviewers", []):
                reviewer = Reviewer(**reviewer_data)
                self.reviewers[reviewer.username] = reviewer

            # Load GitHub token if available
            github_token = config.get("github_token") or os.environ.get("GITHUB_TOKEN")
            if github_token:
                self.github = Github(github_token)

    def _load_assignments(self):
        """Load current assignments."""
        assignments_file = Path("review_assignments.json")
        if assignments_file.exists():
            with open(assignments_file) as f:
                assignments_data = json.load(f)

            self.assignments = []
            for assignment_data in assignments_data:
                assignment = ReviewAssignment(
                    standard_name=assignment_data["standard_name"],
                    reviewer=assignment_data["reviewer"],
                    stage=assignment_data["stage"],
                    assigned_date=datetime.fromisoformat(
                        assignment_data["assigned_date"]
                    ),
                    due_date=datetime.fromisoformat(assignment_data["due_date"]),
                    status=assignment_data["status"],
                    feedback_url=assignment_data.get("feedback_url"),
                )
                self.assignments.append(assignment)

    def _save_assignments(self):
        """Save assignments to file."""
        assignments_data = []
        for assignment in self.assignments:
            assignment_dict = asdict(assignment)
            assignment_dict["assigned_date"] = assignment.assigned_date.isoformat()
            assignment_dict["due_date"] = assignment.due_date.isoformat()
            assignments_data.append(assignment_dict)

        with open("review_assignments.json", "w") as f:
            json.dump(assignments_data, f, indent=2)

    def assign_reviewers(
        self, standard_name: str, domain: str, pr_number: int | None = None
    ) -> dict[str, list[str]]:
        """
        Assign reviewers for a standard based on domain and availability.

        Args:
            standard_name: Name of the standard to review
            domain: Domain/category of the standard
            pr_number: GitHub PR number if applicable

        Returns:
            Dictionary mapping review stages to assigned reviewers
        """
        assignments = {"technical": [], "editorial": [], "community": []}

        # Technical reviewers (domain experts)
        technical_reviewers = self._find_available_reviewers(
            domains=[domain], roles=["maintainer", "domain_expert"], max_assignments=2
        )
        assignments["technical"] = technical_reviewers[
            :2
        ]  # Assign 2 technical reviewers

        # Editorial reviewer
        editorial_reviewers = self._find_available_reviewers(
            roles=["editorial", "maintainer"], max_assignments=1
        )
        assignments["editorial"] = editorial_reviewers[
            :1
        ]  # Assign 1 editorial reviewer

        # Community reviewers
        community_reviewers = self._find_available_reviewers(
            roles=["community", "domain_expert", "maintainer"],
            max_assignments=3,
            exclude=assignments["technical"] + assignments["editorial"],
        )
        assignments["community"] = community_reviewers[
            :3
        ]  # Assign 3 community reviewers

        # Create review assignments
        now = datetime.utcnow()

        # Technical review: 5 days
        for reviewer in assignments["technical"]:
            assignment = ReviewAssignment(
                standard_name=standard_name,
                reviewer=reviewer,
                stage="technical",
                assigned_date=now,
                due_date=now + timedelta(days=5),
                status="assigned",
            )
            self.assignments.append(assignment)
            self.reviewers[reviewer].current_assignments += 1

        # Editorial review: 3 days (starts after technical)
        for reviewer in assignments["editorial"]:
            assignment = ReviewAssignment(
                standard_name=standard_name,
                reviewer=reviewer,
                stage="editorial",
                assigned_date=now + timedelta(days=5),
                due_date=now + timedelta(days=8),
                status="assigned",
            )
            self.assignments.append(assignment)
            self.reviewers[reviewer].current_assignments += 1

        # Community review: 7 days (overlaps with editorial)
        for reviewer in assignments["community"]:
            assignment = ReviewAssignment(
                standard_name=standard_name,
                reviewer=reviewer,
                stage="community",
                assigned_date=now + timedelta(days=3),
                due_date=now + timedelta(days=10),
                status="assigned",
            )
            self.assignments.append(assignment)
            self.reviewers[reviewer].current_assignments += 1

        # Save assignments
        self._save_assignments()

        # Create GitHub assignments if PR number provided
        if pr_number and self.github:
            self._create_github_assignments(pr_number, assignments)

        logger.info(f"Assigned reviewers for {standard_name}: {assignments}")
        return assignments

    def _find_available_reviewers(
        self,
        domains: list[str] | None = None,
        roles: list[str] | None = None,
        max_assignments: int = 5,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Find available reviewers matching criteria."""
        available = []
        exclude = exclude or []

        for username, reviewer in self.reviewers.items():
            if username in exclude:
                continue

            if not reviewer.availability:
                continue

            if reviewer.current_assignments >= reviewer.max_concurrent:
                continue

            if roles and reviewer.role not in roles:
                continue

            if domains and not any(domain in reviewer.domains for domain in domains):
                continue

            available.append(username)

            if len(available) >= max_assignments:
                break

        return available

    def _create_github_assignments(
        self, pr_number: int, assignments: dict[str, list[str]]
    ):
        """Create GitHub review assignments."""
        if not self.github:
            return

        try:
            repo = self.github.get_repo("williamzujkowski/mcp-standards-server")
            pr = repo.get_pull(pr_number)

            # Assign all reviewers to the PR
            all_reviewers = []
            for stage_reviewers in assignments.values():
                all_reviewers.extend(stage_reviewers)

            if all_reviewers:
                pr.create_review_request(reviewers=all_reviewers)
                logger.info(f"Created GitHub review requests for PR #{pr_number}")

        except Exception as e:
            logger.error(f"Failed to create GitHub assignments: {e}")

    def complete_review(
        self,
        standard_name: str,
        reviewer: str,
        stage: str,
        feedback_url: str | None = None,
    ):
        """Mark a review as completed."""
        for assignment in self.assignments:
            if (
                assignment.standard_name == standard_name
                and assignment.reviewer == reviewer
                and assignment.stage == stage
            ):

                assignment.status = "completed"
                assignment.feedback_url = feedback_url

                # Decrease reviewer's current assignments
                if reviewer in self.reviewers:
                    self.reviewers[reviewer].current_assignments = max(
                        0, self.reviewers[reviewer].current_assignments - 1
                    )

                self._save_assignments()
                logger.info(
                    f"Marked review completed: {standard_name} by {reviewer} ({stage})"
                )
                break

    def get_overdue_reviews(self) -> list[ReviewAssignment]:
        """Get list of overdue review assignments."""
        now = datetime.utcnow()
        overdue = []

        for assignment in self.assignments:
            if (
                assignment.status in ["assigned", "in_progress"]
                and assignment.due_date < now
            ):
                assignment.status = "overdue"
                overdue.append(assignment)

        if overdue:
            self._save_assignments()

        return overdue

    def get_reviewer_workload(self) -> dict[str, dict[str, Any]]:
        """Get current workload for all reviewers."""
        workload = {}

        for username, reviewer in self.reviewers.items():
            active_assignments = [
                a
                for a in self.assignments
                if a.reviewer == username and a.status in ["assigned", "in_progress"]
            ]

            workload[username] = {
                "name": reviewer.name,
                "role": reviewer.role,
                "domains": reviewer.domains,
                "current_assignments": len(active_assignments),
                "max_concurrent": reviewer.max_concurrent,
                "availability": reviewer.availability,
                "assignments": [
                    {
                        "standard": a.standard_name,
                        "stage": a.stage,
                        "due_date": a.due_date.isoformat(),
                        "status": a.status,
                    }
                    for a in active_assignments
                ],
            }

        return workload

    def send_reminder_notifications(self):
        """Send reminder notifications for pending reviews."""
        now = datetime.utcnow()

        # Find assignments due within 24 hours
        upcoming_due = []
        for assignment in self.assignments:
            if assignment.status in [
                "assigned",
                "in_progress",
            ] and assignment.due_date - now <= timedelta(hours=24):
                upcoming_due.append(assignment)

        # Send notifications
        for assignment in upcoming_due:
            self._send_notification(assignment, "reminder")

        # Send overdue notifications
        overdue = self.get_overdue_reviews()
        for assignment in overdue:
            self._send_notification(assignment, "overdue")

    def _send_notification(self, assignment: ReviewAssignment, notification_type: str):
        """Send notification to reviewer."""
        reviewer = self.reviewers.get(assignment.reviewer)
        if not reviewer or not reviewer.contact_info:
            return

        # Prepare notification data
        {
            "type": notification_type,
            "reviewer": reviewer.name,
            "standard": assignment.standard_name,
            "stage": assignment.stage,
            "due_date": assignment.due_date.isoformat(),
            "status": assignment.status,
        }

        # Send notification (webhook, email, etc.)
        # This is a placeholder - implement actual notification logic
        logger.info(
            f"Notification sent to {reviewer.name}: {notification_type} for {assignment.standard_name}"
        )

    def generate_review_report(self) -> dict[str, Any]:
        """Generate comprehensive review status report."""
        now = datetime.utcnow()

        # Calculate metrics
        total_assignments = len(self.assignments)
        completed = len([a for a in self.assignments if a.status == "completed"])
        overdue = len([a for a in self.assignments if a.status == "overdue"])
        in_progress = len([a for a in self.assignments if a.status == "in_progress"])

        # Group by standard
        standards = {}
        for assignment in self.assignments:
            if assignment.standard_name not in standards:
                standards[assignment.standard_name] = {
                    "assignments": [],
                    "completed": 0,
                    "total": 0,
                }

            standards[assignment.standard_name]["assignments"].append(
                {
                    "reviewer": assignment.reviewer,
                    "stage": assignment.stage,
                    "status": assignment.status,
                    "due_date": assignment.due_date.isoformat(),
                }
            )
            standards[assignment.standard_name]["total"] += 1

            if assignment.status == "completed":
                standards[assignment.standard_name]["completed"] += 1

        return {
            "timestamp": now.isoformat(),
            "summary": {
                "total_assignments": total_assignments,
                "completed": completed,
                "in_progress": in_progress,
                "overdue": overdue,
                "completion_rate": (
                    completed / total_assignments if total_assignments > 0 else 0
                ),
            },
            "standards": standards,
            "reviewer_workload": self.get_reviewer_workload(),
        }


def main():
    """CLI interface for reviewer tools."""
    parser = argparse.ArgumentParser(description="Reviewer management tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Assign reviewers command
    assign_parser = subparsers.add_parser(
        "assign", help="Assign reviewers to a standard"
    )
    assign_parser.add_argument("--standard", required=True, help="Standard name")
    assign_parser.add_argument("--domain", required=True, help="Standard domain")
    assign_parser.add_argument("--pr", type=int, help="GitHub PR number")

    # Complete review command
    complete_parser = subparsers.add_parser("complete", help="Mark review as completed")
    complete_parser.add_argument("--standard", required=True, help="Standard name")
    complete_parser.add_argument("--reviewer", required=True, help="Reviewer username")
    complete_parser.add_argument("--stage", required=True, help="Review stage")
    complete_parser.add_argument("--feedback-url", help="URL to feedback")

    # Status commands
    subparsers.add_parser("overdue", help="List overdue reviews")
    subparsers.add_parser("workload", help="Show reviewer workload")
    subparsers.add_parser("report", help="Generate comprehensive report")
    subparsers.add_parser("notify", help="Send reminder notifications")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize manager
    manager = ReviewerManager()

    if args.command == "assign":
        assignments = manager.assign_reviewers(
            standard_name=args.standard, domain=args.domain, pr_number=args.pr
        )
        print(f"Assigned reviewers for {args.standard}:")
        for stage, reviewers in assignments.items():
            print(f"  {stage}: {', '.join(reviewers)}")

    elif args.command == "complete":
        manager.complete_review(
            standard_name=args.standard,
            reviewer=args.reviewer,
            stage=args.stage,
            feedback_url=args.feedback_url,
        )
        print(f"Marked review completed: {args.standard} by {args.reviewer}")

    elif args.command == "overdue":
        overdue = manager.get_overdue_reviews()
        if overdue:
            print("Overdue reviews:")
            for assignment in overdue:
                print(
                    f"  {assignment.standard_name} - {assignment.reviewer} ({assignment.stage}) - Due: {assignment.due_date}"
                )
        else:
            print("No overdue reviews")

    elif args.command == "workload":
        workload = manager.get_reviewer_workload()
        print("Reviewer workload:")
        for username, info in workload.items():
            print(
                f"  {info['name']} ({username}): {info['current_assignments']}/{info['max_concurrent']} assignments"
            )

    elif args.command == "report":
        report = manager.generate_review_report()
        print(json.dumps(report, indent=2))

    elif args.command == "notify":
        manager.send_reminder_notifications()
        print("Reminder notifications sent")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
