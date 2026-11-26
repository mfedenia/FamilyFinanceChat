---
name: vm-ssh-access-needs-project-level-read
description: Team SSH to qasap-vm01 needs a project-level compute read role, not just instance-level osAdminLogin; gcloud compute ssh fails on compute.projects.get otherwise.
metadata:
  type: project
---

Team SSH access to `qasap-vm01` (OS Login is on) is granted to `group:fin602-openwebui@g-groups.wisc.edu`:

- instance-level `roles/compute.osAdminLogin` on `qasap-vm01`
- `roles/iam.serviceAccountUser` on the VM's service account `476712090189-compute@developer.gserviceaccount.com`
- **project-level** `roles/compute.viewer` and `roles/iap.tunnelResourceAccessor`

All four carry the condition `vm01-fall26` (expires 2026-12-21).

**Why:** on 2026-11-18 the project-level roles were tightened to instance level only. That broke
`gcloud compute ssh`, which always calls `compute.projects.get` on the *project* — an
instance-level grant cannot satisfy it. Audit logs on 2026-11-22 showed the team denied on
exactly that call. Port 22 is open only to UW campus/VPN ranges and the IAP range, so off
campus a student needs VPN or `--tunnel-through-iap`.

**How to apply:** when onboarding a new team, add them to the group (UW Manifest), extend the
condition dates, and keep the project-level read role. A denial on `compute.instances.get`
(not `projects.get`) means that person isn't in the group, or is signed in to gcloud with a
non-wisc.edu account.
