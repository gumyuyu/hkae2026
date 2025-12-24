const fs = require('fs');
const https = require('https');
const express = require('express');
const path = require('path');

const app = express();
const PORT = 8000;

// Serve static files
app.use(express.static(__dirname));

// HTTPS options
const options = {
  key: fs.readFileSync('key.pem'),
  cert: fs.readFileSync('cert.pem')
};

// Start HTTPS server
https.createServer(options, app).listen(PORT, '0.0.0.0', () => {
  console.log(`HTTPS server running at https://0.0.0.0:${PORT}`);
});
