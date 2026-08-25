from __future__ import annotations

import argparse
import getpass

from webapp.config import Settings
from webapp.db import Database
from webapp.security import hash_password, normalise_email, valid_email


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage LibPrep Cloud")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create or upgrade the application database")
    create = subparsers.add_parser("create-admin", help="Create an approved administrator")
    create.add_argument("--email", required=True)
    create.add_argument("--name", required=True)
    args = parser.parse_args()

    settings = Settings()
    database = Database(settings.database_path)
    database.initialize()
    if args.command == "init-db":
        print(f"Database ready: {settings.database_path}")
        return

    email = normalise_email(args.email)
    name = " ".join(args.name.strip().split())
    if not valid_email(email) or not 2 <= len(name) <= 80:
        raise SystemExit("Invalid administrator name or email")
    if database.get_user_by_email(email):
        raise SystemExit("An account already exists for that email")
    password = getpass.getpass("Administrator password (12+ characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation or not 12 <= len(password) <= 128:
        raise SystemExit("Passwords do not match or are outside the 12 to 128 character limit")
    user = database.create_user(
        email=email, display_name=name, password_hash=hash_password(password),
        role="admin", status="approved",
    )
    database.audit("user.create_admin", "user", user["id"], actor_user_id=user["id"])
    print(f"Administrator created: {email}")


if __name__ == "__main__":
    main()
