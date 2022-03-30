const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  runtimeCompiler: true //Vue报错解决 You are using the runtime-only build of Vue where the template compiler is not available.
})
