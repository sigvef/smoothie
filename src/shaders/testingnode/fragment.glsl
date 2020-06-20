uniform float frame;
uniform float resolutionX;
uniform float resolutionY;
uniform sampler2D iChannel0;

float iTime = frame / 60.;
vec2 iResolution = vec2(resolutionX, resolutionY);

varying vec2 vUv;

void main() {
    gl_FragColor = vec4(vUv, 0.5 + 0.5 * sin(frame / 60.0), 1.0);
}
