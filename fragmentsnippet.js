"char fragment[] = {" +
  (
    "#version 130\n                               " +
    SHADERS.main.fragmentShader
  )
    .split("")
    .map((x) => x.charCodeAt(0))
    .map((x, i) => "0x" + x.toString(16) + (i % 16 === 0 ? "\n" : ""))
    .join(",") +
  ",0x00};";
