#!/usr/bin/env python3
"""
Backup to S3 script
Creates a backup archive of a specified folder and uploads it to S3 with public access.
Works on Windows, macOS, and Linux.
"""

import datetime
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3


def main():
    """Main function to backup a folder to S3"""
    print("🚀 Starting S3 backup process...")

    # Get environment variables with defaults
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    bucket_path = os.environ.get("S3_BUCKET_PATH", "mini-llm-backup/")
    folder_to_backup = os.environ.get("FOLDER_TO_BACKUP", "./temp")

    # Validate required environment variables
    missing_vars = []
    if not aws_access_key:
        missing_vars.append("AWS_ACCESS_KEY_ID")
    if not aws_secret_key:
        missing_vars.append("AWS_SECRET_ACCESS_KEY")
    if not aws_region:
        missing_vars.append("AWS_DEFAULT_REGION")
    if not bucket_name:
        missing_vars.append("S3_BUCKET_NAME")

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print(
            "\nRequired: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET_NAME"
        )
        print(
            "Optional: S3_BUCKET_PATH (default: mini-llm-backup/), FOLDER_TO_BACKUP (default: ./temp)"
        )
        sys.exit(1)

    # Ensure bucket path ends with a trailing slash
    if not bucket_path.endswith("/"):
        bucket_path += "/"

    # Validate folder to backup exists
    folder_path = Path(folder_to_backup)
    if not folder_path.exists():
        print(f"❌ Folder to backup does not exist: {folder_to_backup}")
        sys.exit(1)

    # Generate archive name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_name = f"backup_{timestamp}.tar.gz"

    # Create temporary directory for archive
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = os.path.join(temp_dir, archive_name)

        # Create tar.gz archive
        print(f"📦 Creating archive of {folder_to_backup}...")
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(folder_to_backup, arcname=os.path.basename(folder_to_backup))
        except Exception as e:
            print(f"❌ Failed to create archive: {str(e)}")
            sys.exit(1)

        # Upload to S3
        print(f"☁️ Uploading to S3 bucket {bucket_name}...")
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )

            # Upload file with public-read ACL
            s3_key = f"{bucket_path}{archive_name}"
            s3_client.upload_file(
                archive_path, bucket_name, s3_key, ExtraArgs={"ACL": "public-read"}
            )

            # Generate public URL
            if aws_region == "us-east-1":
                # Special case for us-east-1 region
                public_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            else:
                public_url = (
                    f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
                )

            print(f"✅ Backup uploaded successfully to s3://{bucket_name}/{s3_key}")
            print(f"🌍 Public URL: {public_url}")

        except Exception as e:
            print(f"❌ Failed to upload to S3: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
