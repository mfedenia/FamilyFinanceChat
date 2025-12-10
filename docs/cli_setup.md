# Google Cloud CLI Setup Guide

This guide walks you through installing the Google Cloud CLI and connecting to the QASAP VM.

## Prerequisites

- A Google account with access to the `wsb-hc-qasap-ae2e` project
- macOS, Linux, or Windows machine

---

## Step 1: Install Google Cloud CLI

### macOS

Using Homebrew:

```bash
brew install --cask google-cloud-sdk
```

Or download directly from: https://cloud.google.com/sdk/docs/install

### Linux (Debian/Ubuntu)

```bash
# Add the Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update and install the CLI
sudo apt-get update && sudo apt-get install google-cloud-cli
```

### Windows

Download and run the installer from: https://cloud.google.com/sdk/docs/install

---

## Step 2: Authenticate with Google Cloud

Run the following command to log in:

```bash
gcloud auth login
```

This will open a browser window. Sign in with your `@wisc.edu` account that has access to the project.

After successful authentication, you should see:

```
You are now logged in as [your-email@wisc.edu].
Your current project is [wsb-hc-qasap-ae2e].
```

---

## Step 3: Set the Project (if needed)

If the correct project isn't set automatically:

```bash
gcloud config set project wsb-hc-qasap-ae2e
```

---

## Step 4: Connect to the VM

SSH into the QASAP VM:

```bash
gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e
```

The first time you connect, it will:
1. Generate SSH keys automatically
2. Add your public key to the VM
3. Connect you to the VM

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `gcloud auth login` | Authenticate with Google Cloud |
| `gcloud config set project PROJECT_ID` | Set the active project |
| `gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e` | SSH into the VM |
| `gcloud compute instances list` | List all VMs in the project |
| `gcloud auth list` | Show authenticated accounts |
| `gcloud config list` | Show current configuration |

---

## Troubleshooting

### "Permission denied" when connecting

Make sure your Google account has the **Compute OS Login** or **Compute Instance Admin** role on the project.

### "Could not fetch resource" error

Run `gcloud auth login` again to refresh your credentials.

### Update the CLI

If prompted to update:

```bash
gcloud components update
```

---

## Exiting the VM

To disconnect from the VM:

```bash
exit
```

Or press `Ctrl+D`.
