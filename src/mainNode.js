(function (global) {
  class mainNode extends NIN.ShaderNode {
    constructor(id, options) {
      const shader = { ...SHADERS[options.shader] };
      shader.fragmentShader = options.fragmentPreamble + shader.fragmentShader;
      super(id, { ...options, shader });

      this.title = Loader.loadTexture("res/title.png");
      this.ninjadev = Loader.loadTexture("res/ninjadev.png");

      this.title.magFilter = THREE.LinearFilter;
      this.title.minFilter = THREE.LinearFilter;
      this.ninjadev.magFilter = THREE.LinearFilter;
      this.ninjadev.minFilter = THREE.LinearFilter;
      this.ninjadev.wrapS = THREE.RepeatWrapping;
      this.ninjadev.wrapT = THREE.RepeatWrapping;
    }

    update(frame) {
      this.uniforms.frame.value = frame;
      this.uniforms.resolutionX.value = 16 * GU;
      this.uniforms.resolutionY.value = 9 * GU;
      this.uniforms.title.value = this.title;

      if (frame >= 7578) {
        this.uniforms.title.value = this.ninjadev;
      }

      this.uniforms.ninjadevTriangles.value = [
        /* N */
        new THREE.Vector2(0, 0),
        new THREE.Vector2(1, 0),
        new THREE.Vector2(0, 9),

        new THREE.Vector2(0, 9),
        new THREE.Vector2(1, 9),
        new THREE.Vector2(1, 0),

        new THREE.Vector2(0, 0),
        new THREE.Vector2(3, 9),
        new THREE.Vector2(3, 6),

        new THREE.Vector2(3, 6),
        new THREE.Vector2(1, 0),
        new THREE.Vector2(0, 0),

        new THREE.Vector2(3, 0),
        new THREE.Vector2(4, 0),
        new THREE.Vector2(3, 9),

        new THREE.Vector2(3, 9),
        new THREE.Vector2(4, 9),
        new THREE.Vector2(4, 0),

        /* I */
        new THREE.Vector2(5, 0),
        new THREE.Vector2(6, 0),
        new THREE.Vector2(5, 9),

        new THREE.Vector2(5, 9),
        new THREE.Vector2(6, 9),
        new THREE.Vector2(6, 0),

        /* N */
        new THREE.Vector2(0 + 7, 0),
        new THREE.Vector2(1 + 7, 0),
        new THREE.Vector2(0 + 7, 9),

        new THREE.Vector2(0 + 7, 9),
        new THREE.Vector2(1 + 7, 9),
        new THREE.Vector2(1 + 7, 0),

        new THREE.Vector2(0 + 7, 0),
        new THREE.Vector2(3 + 7, 9),
        new THREE.Vector2(3 + 7, 6),

        new THREE.Vector2(3 + 7, 6),
        new THREE.Vector2(1 + 7, 0),
        new THREE.Vector2(0 + 7, 0),

        new THREE.Vector2(3 + 7, 0),
        new THREE.Vector2(4 + 7, 0),
        new THREE.Vector2(3 + 7, 9),

        new THREE.Vector2(3 + 7, 9),
        new THREE.Vector2(4 + 7, 9),
        new THREE.Vector2(4 + 7, 0),

        /* J */
        new THREE.Vector2(13, 0),
        new THREE.Vector2(14, 0),
        new THREE.Vector2(14, 4),

        new THREE.Vector2(14, 4),
        new THREE.Vector2(13, 4),
        new THREE.Vector2(13, 0),

        new THREE.Vector2(14, 4),
        new THREE.Vector2(12, 10),
        new THREE.Vector2(11, 10),

        new THREE.Vector2(11, 10),
        new THREE.Vector2(13, 4),
        new THREE.Vector2(14, 4),

        /* A */
        new THREE.Vector2(16, 0),
        new THREE.Vector2(19, 9),
        new THREE.Vector2(18, 9),

        new THREE.Vector2(18, 9),
        new THREE.Vector2(16, 3),
        new THREE.Vector2(16, 0),

        new THREE.Vector2(16, 0),
        new THREE.Vector2(13, 9),
        new THREE.Vector2(14, 9),

        new THREE.Vector2(14, 9),
        new THREE.Vector2(16, 3),
        new THREE.Vector2(16, 0),

        /* D */
        new THREE.Vector2(17, 0),
        new THREE.Vector2(21, 0),
        new THREE.Vector2(20, 1),

        new THREE.Vector2(20, 1),
        new THREE.Vector2(18.3333333333, 1),
        new THREE.Vector2(17, 0),

        new THREE.Vector2(17, 0),
        new THREE.Vector2(20, 9),
        new THREE.Vector2(20, 6),

        new THREE.Vector2(20, 6),
        new THREE.Vector2(17, 0),
        new THREE.Vector2(18.3333333333, 1),

        new THREE.Vector2(20, 1),
        new THREE.Vector2(21, 0),
        new THREE.Vector2(21, 6),

        new THREE.Vector2(21, 6),
        new THREE.Vector2(20, 6),
        new THREE.Vector2(20, 1),

        new THREE.Vector2(20, 6),
        new THREE.Vector2(21, 6),
        new THREE.Vector2(20, 9),

        /* E */
        new THREE.Vector2(22, 0),
        new THREE.Vector2(24, 0),
        new THREE.Vector2(23, 1),

        new THREE.Vector2(23, 1),
        new THREE.Vector2(24, 1),
        new THREE.Vector2(24, 0),

        new THREE.Vector2(22, 0),
        new THREE.Vector2(22, 9),
        new THREE.Vector2(23, 9),

        new THREE.Vector2(22, 0),
        new THREE.Vector2(23, 0),
        new THREE.Vector2(23, 9),

        new THREE.Vector2(23, 4),
        new THREE.Vector2(24, 4),
        new THREE.Vector2(24, 5),

        new THREE.Vector2(24, 5),
        new THREE.Vector2(23, 5),
        new THREE.Vector2(23, 4),

        new THREE.Vector2(23, 8),
        new THREE.Vector2(25, 8),
        new THREE.Vector2(25, 9),

        new THREE.Vector2(25, 9),
        new THREE.Vector2(23, 9),
        new THREE.Vector2(23, 8),

        /* V */
        new THREE.Vector2(25, 0),
        new THREE.Vector2(26, 0),
        new THREE.Vector2(26, 6),

        new THREE.Vector2(26, 6),
        new THREE.Vector2(25, 6),
        new THREE.Vector2(25, 0),

        new THREE.Vector2(25, 6),
        new THREE.Vector2(26, 6),
        new THREE.Vector2(26, 9),

        new THREE.Vector2(26, 9),
        new THREE.Vector2(26, 6),
        new THREE.Vector2(28, 0),

        new THREE.Vector2(28, 0),
        new THREE.Vector2(29, 0),
        new THREE.Vector2(26, 9),
      ];
      this.uniforms.ninjadevTrianglesCount.value =
        this.uniforms.ninjadevTriangles.value.length / 3;
    }

    exportImage() {
      const img = this.title.image;
      const c = document.createElement("canvas");
      const x = c.getContext("2d");
      c.width = img.width;
      c.height = img.height;
      x.translate(c.width / 2, c.height / 2);
      x.scale(1, -1);
      x.drawImage(img, -c.width / 2, -c.height / 2);
      const d = x
        .getImageData(0, 0, c.width, c.height)
        .data.filter((x, i) => i % 4 !== 3);
      const pixels = [];
      const map = {};
      [0, 46, 101, 140, 172, 212, 241, 255].map((x, i) => {
        map[`${x},${x},${x}`] = i;
      });
      for (let i = 0; i < d.length; i += 3) {
        pixels.push(map[d.slice(i, i + 3).join(",")]);
      }

      const bits = pixels
        .flatMap((x) => x.toString(2).padStart(3, "0").split(""))
        .map((x) => +x);

      const packed = [];
      for (let i = 0; i < bits.length; i += 8) {
        packed.push(parseInt(bits.slice(i, i + 8).join(""), 2));
      }

      document.body.appendChild(c);

      const output =
        "unsigned char packed[] = {" +
        packed.map((x, i) => x + (i % 16 === 0 ? "\n" : "")) +
        "};";

      window.magicOutput = output;
      console.log(output);
    }
  }

  global.mainNode = mainNode;
})(this);
