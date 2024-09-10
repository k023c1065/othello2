from github import Github
import os

# Replace with your GitHub token
GITHUB_TOKEN = 'your_github_token'
# Replace with your repository name
REPO_NAME = 'your_username/your_repository'
# Path to the file you want to upload
FILE_PATH = 'upload_model.py'
# Commit message
COMMIT_MESSAGE = 'Add newest model file'

# Authenticate to GitHub
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

# Read the file content
with open(FILE_PATH, 'r') as file:
    content = file.read()

# Get the path of the file in the repository
file_path_in_repo = os.path.basename(FILE_PATH)

# Check if the file already exists in the repository
try:
    contents = repo.get_contents(file_path_in_repo)
    # Update the file if it exists
    repo.update_file(contents.path, COMMIT_MESSAGE, content, contents.sha)
    print(f'File {file_path_in_repo} updated in the repository.')
except:
    # Create the file if it does not exist
    repo.create_file(file_path_in_repo, COMMIT_MESSAGE, content)
    print(f'File {file_path_in_repo} created in the repository.')

