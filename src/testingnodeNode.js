(function(global) {
  class testingnodeNode extends NIN.ShaderNode {
    constructor(id, options) {
      super(id, options);
    }

    update(frame) {
      this.uniforms.frame.value = frame;
    }
  }

  global.testingnodeNode = testingnodeNode;
})(this);
