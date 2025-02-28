import { defineConfig } from 'wxt';
import type { UserConfig } from 'wxt';

// @see https://wxt.dev/api/config.html

export default defineConfig({
  extensionApi: 'chrome',
  modules: ['@wxt-dev/module-vue'],
  outDir: 'out',
  manifest: {
    permissions: [
      'activeTab',
      'scripting',
      'aiLanguageModelOriginTrial',
      'tabs',
      'webNavigation',
      'storage',
    ],
    host_permissions: ['<all_urls>'],
  },
} satisfies UserConfig);
