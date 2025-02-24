<script lang="ts" setup>
import { ref } from 'vue';

const logs = ref<string[]>([]);
const error = ref<string>('');

function addLog(message: string) {
  logs.value.push(`[${new Date().toLocaleTimeString()}] ${message}`);
}

function clearLogs() {
  logs.value = [];
  error.value = '';
}

// Example usage
addLog('Popup initialized');
</script>

<template>
  <div class="debug-container">
    <h1>Extension Debugger</h1>
    
    <div class="log-container">
      <div v-for="(log, index) in logs" :key="index" class="log-entry">
        {{ log }}
      </div>
    </div>

    <div v-if="error" class="error-message">
      Error: {{ error }}
    </div>

    <div class="controls">
      <button @click="clearLogs">Clear Logs</button>
      <button @click="addLog('Test log entry')">Add Test Log</button>
    </div>
  </div>
</template>

<style scoped>
.debug-container {
  width: 400px;
  padding: 1rem;
  font-family: monospace;
}

.log-container {
  height: 300px;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 0.5rem;
  margin-bottom: 1rem;
  background: #f5f5f5;
}

.log-entry {
  margin: 0.25rem 0;
  color: #333;
}

.error-message {
  color: #ff0000;
  margin-bottom: 1rem;
}

.controls {
  display: flex;
  gap: 0.5rem;
}

button {
  padding: 0.5rem 1rem;
  background: #4CAF50;
  color: white;
  border: none;
  cursor: pointer;
}

button:hover {
  background: #45a049;
}

.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #54bc4ae0);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
