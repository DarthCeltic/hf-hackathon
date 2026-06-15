# ET-SoC1 Board Access

ET-SoC1 board access is available to hackathon participants through the lab
board pool. Use Discord to request access and coordinate debugging.

## Request Access

1. Join our Discord: <https://discord.gg/CbSA2umxf6>
2. Go to the `#Lab` channel.
3. Ask for ET-SoC1 board access.
4. A lab maintainer will help you join the Tailscale tailnet and confirm that
   you can reach the board pool.

Suggested message:

```text
Hi, I am participating in the ET-SoC1 hackathon and would like board access.

Name:
HF/GitHub handle:
Model or port I am working on:
Timezone:
```

After access is approved, follow [`ET_SOC1_QUICKSTART.md`](ET_SOC1_QUICKSTART.md)
to set up your local toolchain and run your first board test.

## Security

You do not need to put personal Tailscale credentials in a PR. Repository CI
uses the configured board runners; Tailscale access is for interactive testing
and debugging.

Do not post private keys, Tailscale auth keys, SSH keys, or personal tokens in
Discord, GitHub, Hugging Face, or this repository. If a maintainer needs to
verify access, they will ask you to confirm connectivity without sharing
secrets.
