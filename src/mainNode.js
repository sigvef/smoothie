(function(global) {
  class mainNode extends NIN.ShaderNode {
    constructor(id, options) {
      super(id, options);
    }

    update(frame) {
      this.uniforms.frame.value = frame;
      this.uniforms.resolutionX.value = 16 * GU;
      this.uniforms.resolutionY.value = 9 * GU;
    }
  }

  global.mainNode = mainNode;
})(this);
