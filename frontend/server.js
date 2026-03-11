const fs = require('fs');
const https = require('https');
const express = require('express');
const path = require('path');
const WebSocket = require('ws');

const app = express();
const PORT = 8000;

// Serve static files
app.use(express.static(__dirname));

// HTTPS options
const options = {
  key: fs.readFileSync('key.pem'),
  cert: fs.readFileSync('cert.pem')
};

// Shared in-memory world state (resets on server restart)
const worldState = new Map();
let sharedXrMode = 'VR';

function broadcast(message, excludeSocket) {
  const payload = JSON.stringify(message);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN && client !== excludeSocket) {
      client.send(payload);
    }
  });
}

// Start HTTPS + WebSocket server
const server = https.createServer(options, app);
const wss = new WebSocket.Server({ server });

wss.on('connection', socket => {
  // Send current world state on connect
  socket.send(JSON.stringify({
    type: 'init',
    objects: Array.from(worldState.values()),
    xrMode: sharedXrMode
  }));

  socket.on('message', data => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch (err) {
      return;
    }

    if (!msg || !msg.type) return;

    if (msg.type === 'create' && msg.object && msg.object.id) {
      if (!msg.object.source && msg.object.base64) {
        msg.object.source = { type: 'base64', value: msg.object.base64 };
        delete msg.object.base64;
      }
      if (!msg.object.source) return;

      worldState.set(msg.object.id, msg.object);
      broadcast({
        type: 'create',
        object: msg.object,
        senderId: msg.senderId
      }, socket);
      return;
    }

    if (msg.type === 'xr-mode' && (msg.mode === 'VR' || msg.mode === 'AR')) {
      sharedXrMode = msg.mode;
      broadcast({
        type: 'xr-mode',
        mode: sharedXrMode,
        senderId: msg.senderId
      }, socket);
      return;
    }

    if (msg.type === 'replace' && msg.id && msg.source) {
      const existing = worldState.get(msg.id);
      if (existing) {
        existing.source = msg.source;
        if (typeof msg.isLoading === 'boolean') {
          existing.isLoading = msg.isLoading;
        }
        broadcast({
          type: 'replace',
          id: msg.id,
          source: msg.source,
          isLoading: msg.isLoading,
          senderId: msg.senderId
        }, socket);
      }
      return;
    }

    if (msg.type === 'update' && msg.id && msg.transform) {
      const existing = worldState.get(msg.id);
      if (existing) {
        existing.transform = msg.transform;
        broadcast({
          type: 'update',
          id: msg.id,
          transform: msg.transform,
          senderId: msg.senderId
        }, socket);
      }
      return;
    }

    if (msg.type === 'delete' && msg.id) {
      if (worldState.delete(msg.id)) {
        broadcast({
          type: 'delete',
          id: msg.id,
          senderId: msg.senderId
        }, socket);
      }
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`HTTPS server running at https://0.0.0.0:${PORT}`);
});
