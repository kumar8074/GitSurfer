# ===================================================================================
# Project: GitSurfer
# File: app/graphs/prompts.py
# Description: This file contains the system prompts used by the graphs.
# Author: LALAN KUMAR
# Created: [19-05-2025]
# Updated: [19-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

SUMMARIZE_STRUCTURE_SYSTEM_PROMPT = """\
Analyze the following list of file paths and produce a JSON structure that represents the file/folder hierarchy. \
Make sure you DO NOT include ```json``` in your response. \
For each file, include an object with metadata like type and size. \
For directories, include their contents. \

Example output:
{
  "dir1": {
    "file1.txt": {"type": "file", "size": 1234},
    "subdir": {
      "file2.txt": {"type": "file", "size": 5678}
    }
  }
}

FILE TREE:
{tree_text}
"""