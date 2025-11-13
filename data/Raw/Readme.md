Raw Dataset Files
The raw NSL-KDD dataset consists of two main files:

KDDTrain+.txt (Training set: ~125,973 rows × 42 columns)
KDDTest+.txt (Testing set: ~22,544 rows × 42 columns)

These are comma-separated text files (no header row). You can download them directly for your GitHub repo using the URLs below. Note: These files are large (~20-30 MB combined), so consider adding them to a .gitignore or using Git LFS if your repo has size limits. Alternatively, host them separately (e.g., on Google Drive or Zenodo) and reference the links in a README.

Download Training: https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
Download Testing: https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt

To fetch them programmatically (e.g., in your repo's setup script), use these commands in your Python script or bash:
```bash 
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```
Column Names (for Reference)
Apply these when loading into Pandas (as in your script):
pythoncolumns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
