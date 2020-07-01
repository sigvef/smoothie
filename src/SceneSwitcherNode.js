(function (global) {
  class SceneSwitcherNode extends NIN.Node {
    constructor(id) {
      super(id, {
        inputs: {
          kiwi: new NIN.TextureInput(),
          peach: new NIN.TextureInput(),
          grapes: new NIN.TextureInput(),
          raspberry: new NIN.TextureInput(),
          strawberry: new NIN.TextureInput(),
          mandarin: new NIN.TextureInput(),
          banana: new NIN.TextureInput(),
          blender: new NIN.TextureInput(),
          outro: new NIN.TextureInput(),
        },
        outputs: {
          render: new NIN.TextureOutput(),
        },
      });
    }

    beforeUpdate(frame) {
      this.inputs.kiwi.enabled = false;
      this.inputs.peach.enabled = false;
      this.inputs.grapes.enabled = false;
      this.inputs.raspberry.enabled = false;
      this.inputs.strawberry.enabled = false;
      this.inputs.mandarin.enabled = false;
      this.inputs.banana.enabled = false;
      this.inputs.blender.enabled = false;
      this.inputs.outro.enabled = false;

      let selectedScene;
      if (frame >= 7578) {
        selectedScene = this.inputs.outro;
      } else if (frame >= 5115) {
        selectedScene = this.inputs.blender;
      } else if (frame >= 4546) {
        selectedScene = this.inputs.banana;
      } else if (frame >= 3940) {
        selectedScene = this.inputs.mandarin;
      } else if (frame >= 3334) {
        selectedScene = this.inputs.strawberry;
      } else if (frame >= 2727) {
        selectedScene = this.inputs.raspberry;
      } else if (frame >= 2121) {
        selectedScene = this.inputs.grapes;
      } else if (frame >= 1515) {
        selectedScene = this.inputs.peach;
      } else if (frame >= 908) {
        selectedScene = this.inputs.kiwi;
      } else {
        selectedScene = this.inputs.kiwi;
      }

      selectedScene.enabled = true;
      this.selectedScene = selectedScene;
    }

    render() {
      this.outputs.render.setValue(this.selectedScene.getValue());
    }
  }

  global.SceneSwitcherNode = SceneSwitcherNode;
})(this);
