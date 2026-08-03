"""Register the spotgp explorer as a jupyter-server-proxy application.

With this in place, the OSCER OnDemand Jupyter session shows a "spotgp
Explorer" tile in the JupyterLab launcher; clicking it starts Streamlit in
the same SLURM job and opens it in a browser tab, proxied through Jupyter.

Enable it once per user:

    mkdir -p ~/.jupyter
    ln -s /ourdisk/hpc/astrogroup/shared/spotgp-project/scripts/ood/jupyter_server_config.py \\
          ~/.jupyter/jupyter_server_config.py

This requires ``jupyter-server-proxy`` in the environment that runs the
*Jupyter server* — not the notebook kernel. If OSCER's OnDemand Jupyter app
does not have it, use the terminal route instead (see docs/oscer.md); it
needs no extra packages.

``absolute_url: True`` plus a matching ``--base-path`` is the combination
Streamlit needs: jupyter-server-proxy forwards the ``/proxy/<port>`` prefix
untouched, and Streamlit builds its asset and websocket URLs from it.
"""

import os

c = get_config()  # noqa: F821

SPOTGP_ROOT = os.environ.get(
    "SPOTGP_ROOT", "/ourdisk/hpc/astrogroup/shared/spotgp-project")

c.ServerProxy.servers = {
    "spotgp": {
        "command": [
            os.path.join(SPOTGP_ROOT, "scripts", "spotgp_app.sh"),
            "--port", "{port}",
            "--base-path", "{base_url}proxy/{port}",
            "--bind", "127.0.0.1",
        ],
        "absolute_url": True,
        "timeout": 120,
        "launcher_entry": {
            "title": "spotgp Explorer",
            "enabled": True,
        },
    },
}
