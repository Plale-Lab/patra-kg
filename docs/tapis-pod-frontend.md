# Tapis Pod: Patra Frontend

## Bad Gateway (502) fix

If `https://patra.pods.icicleai.tapis.io/` returns **Bad Gateway**, the ingress is forwarding to a port the container does not listen on.

The image `plalelab/patra-frontend` runs **nginx**, which listens on **port 80** by default. The pod config must use **port 80**, not 5000.

### Updated payload

Use the payload in **[patra-pod-payload.json](./patra-pod-payload.json)** when creating or updating the pod. It has `networking.default.port` set to **80**. Omit server-managed fields (`time_to_stop_ts`, `status`, `status_container`, `creation_ts`, `update_ts`, `start_instance_ts`) in update requests.

Then update the pod (e.g. via Tapis Pods API or UI) so the new config is applied. The frontend should then be reachable at `https://patra.pods.icicleai.tapis.io/`.

### If you build the frontend image

To keep using port 5000 in the pod, configure nginx (or the app) to listen on 5000, e.g.:

- **nginx**: set `listen 5000;` in the `server` block and `EXPOSE 5000` in the Dockerfile.
- **Node/Vite**: run the dev/server with `--port 5000` and `EXPOSE 5000`.

Otherwise, using **port 80** in the Tapis pod is the simplest fix for the current image.
