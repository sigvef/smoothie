#pragma optimize(off)

uniform float frame;
uniform float resolutionX;
uniform float resolutionY;

#define EPSILON 0.0001
#define PI 3.14159265359

/* turn on or off 4xAA */
//#define AA

#define M_BG 0.

#define M_STRAWBERRY 2.
#define M_KIWI 3.
#define M_MANDARIN 4.
#define M_MANDARIN_PEEL 5.
#define M_RASPBERRY 6.
#define M_GRAPE 7.
#define M_BLENDER 8.
#define M_BLENDER_GLASS 9.
#define M_BLENDER_GLASS_SMOOTHIE 10.
#define M_BANANA 11.
#define M_PEACH 12.
#define M_PEACH_SEED 13.
#define M_GLASS 14.
#define M_LIQUID_SMOOTHIE 15.
#define M_METAL 16.

#define CHECK_MATERIAL(x, material) (((x) > (material - .5)) && ((x) < (material + .5)))

#define Z(x) (clamp((x), 0., 1.))


 vec3 opTwist(vec3 p, float k)
{
        float c = cos(k*p.y);
            float s = sin(k*p.y);
                mat2  m = mat2(c,-s,s,c);
                    return vec3(m*p.xz,p.y);
}

float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }


    float opSmoothIntersection( float d1, float d2, float k ) {
        float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
        return mix( d2, d1, h ) + k*h*(1.0-h); }


        vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
        vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

        float snoise(vec3 v){ 
            const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
            const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

            // First corner
            vec3 i  = floor(v + dot(v, C.yyy) );
            vec3 x0 =   v - i + dot(i, C.xxx) ;

            // Other corners
            vec3 g = step(x0.yzx, x0.xyz);
            vec3 l = 1.0 - g;
            vec3 i1 = min( g.xyz, l.zxy );
            vec3 i2 = max( g.xyz, l.zxy );

            //  x0 = x0 - 0. + 0.0 * C 
            vec3 x1 = x0 - i1 + 1.0 * C.xxx;
            vec3 x2 = x0 - i2 + 2.0 * C.xxx;
            vec3 x3 = x0 - 1. + 3.0 * C.xxx;

            // Permutations
            i = mod(i, 289.0 ); 
            vec4 p = permute( permute( permute( 
                            i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                        + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
                    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

            // Gradients
            // ( N*N points uniformly over a square, mapped onto an octahedron.)
            float n_ = 1.0/7.0; // N=7
            vec3  ns = n_ * D.wyz - D.xzx;

            vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

            vec4 x_ = floor(j * ns.z);
            vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

            vec4 x = x_ *ns.x + ns.yyyy;
            vec4 y = y_ *ns.x + ns.yyyy;
            vec4 h = 1.0 - abs(x) - abs(y);

            vec4 b0 = vec4( x.xy, y.xy );
            vec4 b1 = vec4( x.zw, y.zw );

            vec4 s0 = floor(b0)*2.0 + 1.0;
            vec4 s1 = floor(b1)*2.0 + 1.0;
            vec4 sh = -step(h, vec4(0.0));

            vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
            vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

            vec3 p0 = vec3(a0.xy,h.x);
            vec3 p1 = vec3(a0.zw,h.y);
            vec3 p2 = vec3(a1.xy,h.z);
            vec3 p3 = vec3(a1.zw,h.w);

            //Normalise gradients
            vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
            p0 *= norm.x;
            p1 *= norm.y;
            p2 *= norm.z;
            p3 *= norm.w;

            // Mix final noise value
            vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
            m = m * m;
            return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                        dot(p2,x2), dot(p3,x3) ) );
        }


mat4 translate(vec3 p) {
    return mat4(
            vec4(1., 0., 0., p.x),
            vec4(0., 1., 0., p.y),
            vec4(0., 0., 1., p.z),
            vec4(0., 0., 0., 1.));
}

vec3 opRepeat(vec3 p, vec3 c) {
    return mod(p+0.5*c,c)-0.5*c;
}

struct MaterialProperties {
    vec3 albedo;
    float roughness;
    float bump;
};

struct Hit {
    float distance;
    float steps;
    vec3 position;
    vec3 uv;
    mat4 transform;
    vec3 albedo;
    float roughness;
    float metalness;
    float emissive;
    float material;
};

float sdCylinder( vec3 p, vec3 c ) {
    return length(p.xz-c.xy)-c.z;
}

float sdSphere( vec3 p, float s ) {
    return length(p)-s;
}

float smosh(float edge, float x, float len) {
    return Z(smoothstep(0., 1., (x - edge) / len));   
}
vec2 hash( vec2 p ) { p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*18.5453); }

vec3 hash3( vec3 p ) { return vec3(hash(p.yz), hash(vec2(p.x + 1000., 0.)).x); }


vec3 skybox(vec3 p) {
    float angle = atan(p.z, p.x) + PI / 3.;
    float amount = Z(mix(0., 1., smosh(-1., p.y + 0.3 * sin(angle), 2.)));
    vec3 color = mix(vec3(33., 25., 3.) / 255., vec3(92., 54., 6.) / 255., smosh(0., amount, 1. / 3.));
    color = mix(color, vec3(182., 215., 224.) / 255., smosh(1. / 3., amount, 1. / 3.));
    color = mix(color, vec3(1.), smosh(2. / 3., amount, 1. / 3.));
    return color;
}



float beckmannDistribution(float x, float roughness) {
    float NdotH = max(x, 0.0001);
    float cos2Alpha = NdotH * NdotH;
    float tan2Alpha = (cos2Alpha - 1.0) / cos2Alpha;
    float roughness2 = roughness * roughness;
    float denom = 3.141592653589793 * roughness2 * cos2Alpha * cos2Alpha;
    return exp(tan2Alpha / roughness2) / denom;
}


float beckmannSpecular(
        vec3 lightDirection,
        vec3 viewDirection,
        vec3 surfaceNormal,
        float roughness) {
    return beckmannDistribution(dot(surfaceNormal, normalize(lightDirection + viewDirection)), roughness);
}

float cookTorranceSpecular(
        vec3 lightDirection,
        vec3 viewDirection,
        vec3 surfaceNormal,
        float roughness,
        float fresnel) {

    float VdotN = max(dot(viewDirection, surfaceNormal), 0.0);
    float LdotN = max(dot(lightDirection, surfaceNormal), 0.0);

    //Half angle vector
    vec3 H = normalize(lightDirection + viewDirection);

    //Geometric term
    float NdotH = max(dot(surfaceNormal, H), 0.0);
    float VdotH = max(dot(viewDirection, H), 0.000001);
    float x = 2.0 * NdotH / VdotH;
    float G = min(1.0, min(x * VdotN, x * LdotN));

    //Distribution term
    float D = beckmannDistribution(NdotH, roughness);

    //Fresnel term
    float F = pow(1.0 - VdotN, fresnel);

    //Multiply terms and done
    return  G * F * D / max(3.14159265 * VdotN * LdotN, 0.000001);
}

vec3 opTx( vec3 p, mat4 m )
{
    return (vec4(p, 1.) * m).xyz;
}

mat4 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
            vec4(1, 0, 0, 0),
            vec4(0, c, -s, 0),
            vec4(0, s, c, 0),
            vec4(0, 0, 0, 1)
            );
}


mat4 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
            vec4(c, 0, s, 0),
            vec4(0, 1, 0, 0),
            vec4(-s, 0, c, 0),
            vec4(0, 0, 0, 1)
            );
}

mat4 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
            vec4(c, -s, 0, 0),
            vec4(s, c, 0, 0),
            vec4(0, 0, 1, 0),
            vec4(0, 0, 0, 1)
            );
}


float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}


float sdCappedCylinder( vec3 p, float h, float r )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}


float sdRoundCone( vec3 p, float r1, float r2, float h )
{
    vec2 q = vec2( length(p.xz), p.y );

    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));

    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;

    return dot(q, vec2(a,b) ) - r1;
}


vec3 fancyLighting(Hit hit, vec3 lightDirection, vec3 normal, vec3 rayDirection, float shadow) {
    float angle = Z(dot(normal, lightDirection));

    float specular = cookTorranceSpecular(
            normalize(lightDirection), 
            normalize(rayDirection), 
            normalize(normal), 
            max(hit.roughness, 0.001),
            mix(0.04, 1., hit.metalness));

    float lighting = angle;
    lighting *= shadow;
    lighting = Z(lighting);

    vec3 diffuse = hit.albedo * lighting;

    return diffuse * (1. - specular) + specular * angle + hit.albedo * hit.emissive;
}

Hit add(Hit a, Hit b) {
    if(a.distance < b.distance) {
        return a;   
    }
    return b;
}
Hit sub(Hit b, Hit a) {
    if(-a.distance > b.distance) {
        a.distance = -a.distance;
        return a;   
    }
    return b;
}

Hit mul(Hit a, Hit b) {
    a.distance = max(a.distance, b.distance);


    return a;
}


float smin( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*h*k*(1.0/6.0);
}

Hit smadd(Hit a, Hit b, float k) {
    float m = smin(a.distance, b.distance, k);

    float t = clamp((a.distance - b.distance) / k * 4., -1., 1.);

    t += 1.;
    t *= 0.5;

    a.distance = m;
    a.albedo = mix(a.albedo, b.albedo, t);
    a.roughness = mix(a.roughness, b.roughness, t);
    a.metalness = mix(a.metalness, b.metalness, t);
    a.emissive = mix(a.emissive, b.emissive, t);                   
    a.material = mix(a.material, b.material, t);

    return a;
}


float sdEgg(vec2 p, float ra, float rb )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x);
    float r = ra - rb;
    return ((p.y<0.0)       ? length(vec2(p.x,  p.y    )) - r :
            (k*(p.x+r)<p.y) ? length(vec2(p.x,  p.y-k*r)) :
            length(vec2(p.x+r,p.y    )) - 2.0*r) - rb;
}





float leaf(vec3 p) {
    float d = sdEgg(p.xy, .2, -1.);
    float h = .0001;
    vec2 w = vec2(d, abs(p.z) - h);
    return min(max(w.x,w.y),0.0) + length(max(w,0.0)) - .005;

}


float sdBox( vec3 p, vec3 b )
{
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec3 calculateLightPosition() {
    return vec3(.0, 3., 0.);
}

float repeat(float x, float r) {
    return mod(x + r * 0.5, r) - r * 0.5;
}


float CubicInterpolate(
        float y0,float y1,
        float y2,float y3,
        float mu)
{
    float a0,a1,a2,a3,mu2;

    mu2 = mu*mu;
    a0 = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
    a1 = y0 - 2.5*y1 + 2.*y2 - 0.5*y3;
    a2 = -0.5*y0 + 0.5*y2;
    a3 = y1;

    return(a0*mu*mu2+a1*mu2+a2*mu+a3);
}


float sdPentagon( vec2 p, float r ) {
    const vec3 k = vec3(0.809016994,0.587785252,0.726542528);
    p.x = abs(p.x);
    p -= 2.0*min(dot(vec2(-k.x,k.y),p),0.0)*vec2(-k.x,k.y);
    p -= 2.0*min(dot(vec2( k.x,k.y),p),0.0)*vec2( k.x,k.y);
    p -= vec2(clamp(p.x,-r*k.z,r*k.z),r);    
    return length(p)*sign(p.y);
}


const mat2 myt = mat2(.12121212, .13131313, -.13131313, .12121212);
const vec2 mys = vec2(1e4, 1e6);

vec2 rhash(vec2 uv) {
    uv *= myt;
    uv *= mys;
    return fract(fract(uv / mys) * uv);
}

vec3 hash(vec3 p) {
    return fract(
            sin(vec3(dot(p, vec3(1.0, 57.0, 113.0)), dot(p, vec3(57.0, 113.0, 1.0)),
                    dot(p, vec3(113.0, 1.0, 57.0)))) *
            43758.5453);
}

vec2 voronoi3(vec3 x, float scale) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    float id = 0.0;
    vec2 res = vec2(100.0);
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                vec3 b = vec3(float(i), float(j), float(k));
                vec3 r = vec3(b) - f + hash(p + b) * scale;
                float d = dot(r, r);

                float cond = max(sign(res.x - d), 0.0);
                float nCond = 1.0 - cond;

                float cond2 = nCond * max(sign(res.y - d), 0.0);
                float nCond2 = 1.0 - cond2;

                id = (dot(p + b, vec3(1.0, 57.0, 113.0)) * cond) + (id * nCond);
                res = vec2(d, res.x) * cond + res * nCond;

                res.y = cond2 * d + nCond2 * res.y;
            }
        }
    }

    return res;
}

float bubble(vec3 x) {
    vec2 res = voronoi3(x, 1.);
    return pow(1. - res.x, 4.);
}


float sdTriPrism( vec3 p, vec2 h )
{
      vec3 q = abs(p);
        return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

vec3 opCheapBend(  vec3 p, float k )
{
    float c = cos(k*p.x);
    float s = sin(k*p.x);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xy,p.z);
    return q;
}


float sdWedge(vec3 p) {
    p = opCheapBend(p, .25);
    p.xyz = p.zyx;

    vec3 boxp = p;
    float d = opSmoothIntersection(sdSphere(p, 1.), sdTriPrism(boxp + vec3(0., 2.5, 0.), vec2(1., 1.)) - .1, .5);
    d -= 0.5;
    return d;
}

Hit kiwihalf(vec3 p, mat4 transform, float t, float slicer) {

    p = opTx(p, transform);
    vec3 op = p;
    float scale = 1.;


    float len = length(p);
    p.x /= scale;

    float r = 1.;
    float angle = atan(p.y, p.x);

    float period = 2.;

    float sliceSize = 1. / 5. * 2.;

    float sliceWidths[2];
    sliceWidths[0] = sliceSize * slicer;
    sliceWidths[1] = sliceSize * slicer;
    float sliceLocations[2];
    sliceLocations[0] = -1. / 3.;
    sliceLocations[1] = -2. / 3.;


    for(int i = 0; i < 2; i++) {

        if(p.x < sliceLocations[i]) {
            p.x += sliceWidths[i];

        }
    }

    float d = scale * sdSphere(p, r);

    float c = 0.;
    for(int i = 0; i < 2; i++) {
        float w = sliceWidths[i] - 0.;
        d = max(d, -sdBox(op - vec3(-c + sliceLocations[i] - w * 0.5, 0., 0.), vec3(w * 0.5, 2., 2.)));
        c += w;
    }



    d = max(d, -sdBox(op - vec3(10., 0., 0.), vec3(10.)));



    Hit result = Hit(d, 0., p, p, transform, vec3(1.), 1., 0., 0., M_KIWI);

    result.distance -= 0.005 * sin(p.x * 20.) * cos(p.x * 12.) * sin(p.y * 15.);
    return result;
}


float sdMandarinPeel(vec3 p, float amount) {
    p = opTwist(p, PI * 2.);

    float d = sdSphere(p, 1.);
    d = max(d, - sdSphere(p, 0.9));
    d = opSmoothSubtraction(sdBox(p + vec3(0.,1.0 + 2. * (1. - amount),0.), vec3(2.)), d, .1);

    d *= 0.25;
    return d;
}

float sdHorseshoe( vec2 p, vec2 c, float r, vec2 w ) {
    p.x = abs(p.x);
    float l = length(p);
    p = mat2(-c.x, c.y, 
            c.y, c.x)*p;
    p = vec2((p.y>0.0)?p.x:l*sign(-c.x),
            (p.x>0.0)?p.y:l );
    p = vec2(p.x,abs(p.y-r))-w;
    return length(max(p,0.0)) + min(0.0,max(p.x,p.y));
}

vec2 opRevolution(vec3 p, float o) {
    vec2 q = vec2( length(p.xz) - o, p.y );
    return q;
}


vec3 opRepLim(vec3 p, float c, vec3 l)
{
    vec3 q = p-c*clamp(floor(p/c + 0.5),-l,l);
    return q;
}


#ifdef IS_BLENDER
Hit mapBlender(vec3 p) {
    float rotation = frame * 0.01;
    vec3 np = p;
    vec3 xxp = p;

    mat4 transform = translate(vec3(0.));

    p.y -= 1.;

    float h = (1. - p.y) * 0.5;


    float r = mix(0.8, 1., h);

    float angle = atan(p.z, p.x);


    r = mix(r, 0., smoothstep(0., 1., smosh(0.95, h, 0.15)));
    r = mix(0.8, r, smoothstep(0., 1., smosh(0.65, h, 0.15)));
    r = mix(0., r, smoothstep(0., 1., smosh(0.55, h, 0.08)));

    r = mix(.7, r, smoothstep(0., 1., smosh(0.48, h, 0.2)));
    r = mix(1. + 0.07 * pow(sin(angle * 3.5), 12.), r, smoothstep(0., 1., smosh(0.1, h, 0.6)));
    r = mix(1., r, smoothstep(0., 1., smosh(0.1, h, 0.2)));
    r = mix(1.15, r, smoothstep(0., 1., smosh(0.1, h, 0.01)));
    r = mix(0., r, smoothstep(0., 1., smosh(0., h, 0.1)));
    r = mix(0., r, Z(smosh(0.025, h, 0.01)));


    r *= 0.3;

    vec3 albedo = vec3(0.);
    float roughness = 0.;
    float metalness = 0.;
    float material = M_BLENDER;

    if(h > .95) {
        albedo = vec3(0.);
        roughness = 0.7;
    } else if(h > 0.75) {
        albedo = vec3(214., 208., 182.) / 255.;
        roughness = 1.;
    } else if(h > 0.58) {
        albedo = vec3(0.5);
        roughness = 0.5 ;
        r += sin(angle * 500.) * 0.00001;
        metalness = 1.;
    } else if(h > 0.108) {
        metalness = 0.;
        roughness = 0.;
        albedo = vec3(1.);
        material = M_BLENDER_GLASS_SMOOTHIE;
        material = M_GLASS;
    } else {
        albedo = vec3(0.);
        roughness = .7;
        metalness = 0.;
    }

    float d = sdCappedCylinder(p, r, 1.);

    float containerd = d;

    d = abs(d) - 0.005;
    d *= 0.5;

    //d = max(d, -sdBox(p - vec3(0., 2., 0.), vec3(2.)));


    Hit result = Hit(d, 0., np, p, transform, albedo, roughness, metalness, 0., material);

    vec3 knifep = p;
    knifep += vec3(0., 0.15, 0.);
    knifep = opTx(knifep, rotateY(+frame * 0.1));
    knifep = opTx(knifep, rotateZ(-sign(knifep.x) * PI * 0.25));
    knifep = opTx(knifep, rotateX(PI / 2.));
    float knifed = sdBox(knifep, vec3(.15, 0.03, 0.002));
    result = add(result, Hit(knifed, 0., p, p, transform, vec3(0.5), 0.1, 1., 0., M_METAL));


    float smooshbase = smosh(5153., frame, (7578. - 5153.) / 2.);
    float smoosher = smooshbase * 1. / 3. + .5;
    Hit food = Hit(9999., 0., p, p, transform, vec3(0.), 0.2, 0., 0., M_BG);
    for(int i = 0; i < 7; i++) {
        vec3 fp = p;
        p.y -= 0.2 * length(p.xz);
        fp = opTx(fp, rotateY(frame * 0.02 + frame * (2. - p.y) * 0.001 + PI * 2. / 7. * float(i)));
        fp = opTx(fp, rotateZ(2. * PI * 2. / 7. * float(i) + frame * 0.1));
        fp = fp + vec3(0.2, 0., 0.);
        vec3 albedo = vec3(1.);
        if(i == 0) {
            /* kiwi green */
            albedo = vec3(185., 219., 106.) / 255.;
        } else if(i == 1) {
            /* peach yellow */
            albedo = vec3(252., 198., 66.) / 255.;
        } else if(i == 2) {
            /* grape green */
            albedo = vec3(222., 232., 134.) / 255.;
        } else if(i == 3) {
            /* raspberry red */
            albedo = vec3(249., 27., 70.) / 255.;
        } else if(i == 4) {
            /* strawberry red */
            albedo = vec3(222., 71., 65.) / 255.;
        } else if(i == 5) {
            /* mandarin orange */
            albedo = vec3(247., 181., 32.) / 255.;
        } else if(i == 6) {
            /* banana yellow */
            albedo = vec3(251., 228., 174.) / 255.;
        }
        albedo = mix(albedo, vec3(207., 99., 185.) / 255., smooshbase);
        food = smadd(food, Hit(sdSphere(fp, .1), 0., p, p, transform, albedo, 0.2, 0., 0., M_BG), smoosher);
    }
    food.distance = max(food.distance, containerd + 0.01);
    result = add(result, food);

    xxp.x += frame * 0.02 + 10. * smooshbase;
    xxp.y += 0.2;
    xxp = opRepLim(xxp, 2., vec3(1000., 0., 0.));
    vec3 bg = vec3(125., 130., 160.) / 255.;
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 8.8), vec3(.5, 10., 1.)) - 0.01, 0., xxp, xxp, translate(vec3(0.)), bg * 1.0 + 0.2, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(1., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg * 1.1, 1., 0., 0., M_BG));

    return result;

}
#endif



Hit mapRaspberry(vec3 p) {
    vec3 xxp = p;
    mat4 transform = translate(vec3(0.));

    float scale = 2.8 + 0.02 * length(10. + sin(p * 5.));

    float ender = smosh(3296., frame, 3334. - 3296.);
    p.y += ender * 2.;

    p *= scale;
    float xId = floor(p.x * 0.25  + 0.5) + 1.;
    float xTime = frame * 190. / 60. / 60. / 4. - xId * 0.05;
    float xTimeId = floor(xTime);
    float t = mod(xTime, 1.);

    t *= smosh(35., xTimeId, 0.001);

    p = opTx(p, rotateX(.2));

    p.y += -2. + 2. / (0.1 + abs(sin(PI * t)));

    p = opRepLim(p, 4., vec3(1., 0., 0.));

    p = opTx(p, rotateY(2. * PI * hash(vec3(21.132 + 500. * xTimeId + 123. * xId)).x + t * 2.));
    p = opTx(p, rotateX(2. * PI * hash(vec3(442.123 + 200. * xTimeId + 421. * xId)).y + t * 5.));

    vec3 op = p;

    float ngl = atan(p.z, p.x);
    vec2 res = voronoi3(vec3(ngl, p.y - 0.05, 0.) * 2.3, 0.333);



    p.y -= p.y * 0.2;
    vec2 q = opRevolution(p, 0.);

    float angle = 0.3;
    vec2 c = vec2(sin(angle), cos(angle));
    float bob = pow(res.x, 2.);
    float maxbob = bob;
    bob *= 0.3;
    float d = sdHorseshoe(q, c, 1., vec2(.5 - bob, .1 - bob));
    d -= 0.15;

    d /= scale;
    Hit result = Hit(d, 0., p, p, transform, vec3(1.), 0., maxbob, 0., M_RASPBERRY);

    vec3 bg = vec3(92., 17., 52.) / 255. * 1.5 + 0.3;
    vec3 bg2 = vec3(92., 17., 52.) / 255. * 1.1;
    vec3 bg3 = vec3(207., 249., 122.) / 255. * 1.05 * 0.2;
    bg = mix(bg, bg3, ender);
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 7.), vec3(10., mix(1.5, 2.5, ender), 1.)), 0., xxp, xxp, translate(vec3(0.)), bg, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg2, 1., 0., 0., M_BG));
    return result;
}


#ifdef IS_STRAWBERRY
Hit mapStrawberry(vec3 p) {

    vec3 xxp = p;
    float ender = smosh(3902., frame, 3940. - 3902.);

    float scale = 2. - ender * 0.2;
    p *= scale;

    float lockY = p.y;

    float f = frame + 5.;
    float timer = 2. * f * 190. / 60. / 60. + sin(2. * PI * frame * 190. / 60. / 60. / 4.);
    p.y += timer + 3.2;
    p.y += ender * 0.5;

    float id = floor(p.y / 8. + 0.5);

    p = opRepLim(p, 8., vec3(0., 999999., 0.));

    vec3 np = p;
    mat4 transform = translate(vec3(0.));


    transform *= rotateY(timer);

    p = opTx(p, transform);


    float len = length(p.zx);
    float angle = atan(p.z, p.x);

    p.y -= sin(len * 0.5) * 0.5;

    p.y *= 1.  + 0.25 * sin(p.y);


    float r = 1.;

    float speeder = 20.;
    float angler = 32.;
    float dimpler = 0.5 * ((sin(angle * angler + np.y * speeder + sin(angle * 8. + PI)) + sin(angle * angler - np.y * speeder)));

    dimpler *= 1. - smosh(0.6, np.y, 0.5);


    r += 0.012 * dimpler;

    float seeds = pow(Z(max(-dimpler, 0.) - 0.85), 0.5) * 2.;

    r += 0.02 * seeds;

    vec3 color = vec3(222., 71., 65.) / 255. * (1. - 0.1 + dimpler * 0.1);


    float variation = max(0., pow(0.5 + 0.5 * sin(angle + 2. * PI * (0.5 + 0.5 * sin(p.y + 2.))), 2.) - 0.5);

    variation += 4. * smosh(0.5, np.y, 0.5);

    color = mix(color, vec3(1.), variation * 0.25);

    color = mix(color, 0.9 * vec3(236., 200., 125.) / 255., smosh(0.2, seeds, 0.01));

    float roughness = 0.2 * max(.5 + variation, seeds);

    float d = sdSphere(p, r);

    d = opSmoothSubtraction(sdBox(p + vec3(0., -2.7, 0.), vec3(2.)), d, 0.05);

    vec2 idhash = fract(hash(vec2(id + 223., id + 131.123)));
    vec3 boxp = opTx(p, rotateZ(0.5 * sin(idhash.x * 1571.1)));
    //boxp = opTx(boxp, rotateY(2. * PI * idhash.y));
    boxp += vec3(1.5, 0., 0.);
    d = opSmoothSubtraction(sdBox(boxp, vec3(1.5)), d, 0.05);

    d /= scale;

    color = pow(color, vec3(1.2));

    Hit result = Hit(d, 0., np, p, transform, color, roughness, 0., 0., M_STRAWBERRY);

    float starter = smosh(3334., frame, 3353. - 3334.) * 1.1 - 0.1;
    vec3 bg = vec3(207., 249., 122.) / 255. * 1.05;
    vec3 bg2 = bg * 0.2;
    vec3 bg3 = vec3(223., 245., 191.) / 255.;
    bg = mix(bg, bg3, ender);
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 7.), vec3(1.9 * starter + 2.2 * ender, 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg2, 1., 0., 0., M_BG));

    //result.distance = max(result.distance, -sdBox(p + vec3(2., 0., 0.),  vec3(2.)));
    return result;
}
#endif


#ifdef IS_MANDARIN
Hit mapMandarin(vec3 p, float isShadowMap) {

    vec3 nnp = p;
    vec3 xxp = p;
    float t = mod(frame * 190./ 60. / 60.  /4. / 4. * 2., 1.);

    float scale = 2.3;

    p *= scale;

    p += vec3(0., 1., 0.);
    p += vec3(0., -5., 0.) * (1. - smosh(0., t, 0.1));
    nnp += vec3(0., -5., 0.) * (1. - smosh(0., t, 0.1)) / scale;

    vec3 np = p;

    mat4 transform = translate(vec3(0.));

    transform *= rotateZ(PI / 2.);
    transform *= rotateX(t);

    p = opTx(p, transform);

    Hit result = Hit(sdSphere(p, 0.), 0., p, p, transform, vec3(1.), 0., 0., 0., M_MANDARIN);

    float danceAngleAmount = smosh(0.25, t, 0.25); 

    float fallAmount = smosh(0.75, t, 0.5);

    float mandarinPeelAmount = -0.1 + 1.1 * smosh(0.15, t, 0.2);

    for(int i = 0; i < 8; i++) {
        fallAmount *= 1. + float(i) * 0.01;
        transform *= rotateX(1. / 8. * PI * 2.);
        mat4 lt = transform * rotateZ(-fallAmount * 3. -danceAngleAmount * (0.5 + 0.5 * sin(float(i) * 0.5 + t * 12.)));
        vec3 xp = opTx(np + vec3(0., 5., 0.) * fallAmount, lt);
        xp += vec3(1., 0., 0.);
        xp += vec3(0., 10., 0.) * fallAmount;
        float d = sdWedge(xp);
        result = add(result, Hit(d, 0., xp, xp, lt, vec3(1.), 0., 0., 0., M_MANDARIN));
    }



    result.distance /= scale;

    if(mandarinPeelAmount < 0.999 && isShadowMap < 0.5) {
        float scale = 1.42 - (mandarinPeelAmount * .333);
        nnp *= scale;
        mat4 peelT = rotateY(-t);
        nnp = opTx(nnp, peelT);
        Hit peel = Hit(sdMandarinPeel(nnp, mandarinPeelAmount), 0., nnp, nnp, peelT, vec3(1.), 0., 0., 0., M_MANDARIN_PEEL);
        peel.distance /= scale;
        result = add(peel, result);
    }

    float starter = smosh(3940., frame, 3959. - 3940.);
    float ender = smosh(4508., frame, 4546. - 4508.);
    vec3 bg = vec3(223., 245., 191.) / 255.;
    vec3 bg2 = bg * 0.6;
    vec3 bg3 = vec3(103., 202., 230.) / 255. * 0.5;
    bg = mix(bg, bg3, ender);

    vec3 xxpr = opTx(xxp, rotateZ(-PI / 4. * 2. / 3.));
    result = add(result, Hit(sdBox(xxpr - vec3(0., 0., 7.), vec3(10., mix(4., 3.2, starter * (1. - ender)), 1.)), 0., xxpr, xxpr, translate(vec3(0.)), bg, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg2, 1., 0., 0., M_BG));
    return result;

    return result;
}
#endif

#ifdef IS_PEACH
Hit mapPeach(vec3 p) {

    float scale = 1.25;
    p *= scale;

    vec3 xxp = p;

    mat4 transform = translate(vec3(0.));
    transform *= rotateY(-0.1 + frame * 0.02);
    transform *= rotateX(0. + frame * 0.02);

    float t = mod((frame + 1.) * 190. / 60. / 60. / 4. / 2., 1.);

    float dropT = smosh(-0.25, t, 0.5);
    float chunkT = smosh(0.25, t, 0.15);
    float chunkT2 = smosh(0.5, t, 0.15);
    float outT = smosh(0.85, t, 0.30);

    p.x += mix(-4.5 * 2., 0., dropT);
    p.x += mix(0., 4.5 * 2., outT);
    p.x += 0.1;

    p = opTx(p, transform);

    p.y *= .9;

    float angle = atan(p.z, p.x);
    float r = .5 - 0.2 * pow(angle / PI, 2.);
    float r2 = .75 * pow(length(p.xz), .1);
    float d = sdRoundedCylinder(p, r, r2, 0.);

    float uvd = d;
    d = opSmoothSubtraction(sdBox(p + vec3(2., 2., 0.), chunkT * vec3(2.02, 2.02, 2.02)), d, 0.03);
    d = opSmoothSubtraction(sdBox(p - vec3(2., 2., 0.), chunkT * vec3(2.02, 2.02, 2.02)), d, 0.03);
    d = opSmoothSubtraction(sdBox(p - vec3(0., 2., 2.), chunkT2 * vec3(2.02, 2.02, 2.02)), d, 0.03);
    d = opSmoothSubtraction(sdBox(p + vec3(0., 2., 2.), chunkT2 * vec3(2.02, 2.02, 2.02)), d, 0.03);

    uvd = smin(uvd, sdSphere(p, .95), 0.1);

    Hit result = Hit(d, 0., p, p * ((1. - r) * 0.5 + (1. - r2) * 0.5), transform, vec3(1.), 0., 0., 0., M_PEACH);

    vec3 stoneP = p;

    float noiz = snoise(p * 10.);

    float junknoiz = snoise(2. + p * 4.);

    noiz = pow(noiz, .8) * 1.;
    junknoiz = pow(junknoiz, .8) * 1.;

    float squasher = 1.25;

    stoneP.x *= squasher ;
    stoneP.z *= squasher ;
    float d2 = sdSphere(stoneP, .25 + 0.02 * noiz);
    d2 /= squasher;

    result = smadd(result, Hit(d2, 0., stoneP, stoneP, transform, vec3(1.), 0., 0., 0., M_PEACH_SEED), 0.05);

    vec3 bg = vec3(103., 202., 230.) / 255. * 0.5;
    bg = Z(bg * 1.1 + 0.3);
    vec3 bg2 = vec3(103., 202., 230.)/ 255. * 0.5 * 1.1;

    float starter = smosh(1515., frame, 1534. - 1515.) * 1.1 - 0.1;
    float ender = smosh(2102., frame, 2121. - 2102.) * 1.1 - 0.1;

    vec3 bg3 = vec3(119., 74., 142.) / 255.;
    vec3 bg4 = vec3(62., 18., 85.) / 255.;

    bg = mix(bg, bg3, ender);
    bg2 = mix(bg2, bg4, ender);

    float h = mix(1.9 * starter, 1.7, ender);

    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 7.), vec3(10., h, 1.)), 0., xxp, xxp, translate(vec3(0.)), bg, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg2, 1., 0., 0., M_BG));

    result.distance /= scale;
    return result;
}
#endif

#ifdef IS_BANANA
Hit mapBanana(vec3 p) {

    vec3 xxp = p;

    float t = mod(frame * 190. / 60. / 60. / 4. / 2., 1.);


    float dropT = smosh(0., t, 0.1);
    float leaveT = smosh(.8, t, 0.2);
    float rotateT = smosh(0.25 - 0.15, t, 0.15);

    float elongateT = smosh(0.5 - 0.15, t, 0.15);

    p += vec3(.09, .13, 1.);

    p += vec3(0., -.08 * elongateT, 0.);

    p += vec3(0., -3. + dropT * 3. + leaveT * 2., elongateT * 6.);

    float scale = 8.;
    p *= scale;
    mat4 transform = translate(vec3(0.));
    transform *= rotateY(mix(0., PI, rotateT));
    transform *= rotateY(mix(0., -0.3, elongateT));

    p = opTx(p, transform);

    vec3 op = p;

    p = opCheapBend(p, 0.03 + p.x * 0.0001);
    p = opTx(p, rotateY(PI / 2.));

    float h = 12.;
    float r = 0.;

    float bananaT = (h + p.z) / (h * 2.);
    p = opTx(p, rotateZ(elongateT * t * 0.2 * (2. * (mod(floor(100. + p.z / 1.5), 2.) - .5 ) * floor(10. + p.z / 1.5) * smosh(0.38, bananaT, 0.01) * (1. - smosh(0.68, bananaT, 0.01)))));


    r = 0.2 * smosh(0., bananaT, .01);
    r = mix(r, .002, smosh(.03, bananaT, .04));
    r = mix(r, .01, smosh(.03, bananaT, .15));
    r = mix(r, 1., smosh(.11, bananaT, .18));
    r = mix(r, 0., smosh(.85, bananaT, .15));


    float peelT = smosh(0.5, rotateT, 0.01);
    float peelTSizer = peelT * 0.13;
    peelTSizer *= smosh(0.35, bananaT, 0.01);
    peelTSizer *= 1. - smosh(0.7, bananaT, 0.01);
    float g = sdPentagon(p.xy, r * 1.1 + 0.02 - peelTSizer * 4.);
    g = smin(g, length(p.xy) - r * 1.2 + peelTSizer, 1.5);
    vec2 w = vec2(g, abs(p.z) - h);
    float d = min(max(w.x, w.y), 0.) + length(max(w, 0.));


    for(int i = -3; i < 3; i++) {
        d = opSmoothSubtraction(sdBox(p + vec3(0., 0., float(i) * 1.5), vec3(5., 5., elongateT)), d, 0.05);
    } 

    d /= scale;
    float dSubber = 0.08 * (pow(r, .125 / 4.));
    d -= dSubber;

    d *= 0.5;

    vec3 uv = vec3(p.xy * (0.78 + min(g, .1)), bananaT);
    uv.xy *= mix(2.5, 1., smosh(0.25, bananaT, 0.1));
    uv.xy *= mix(1., 2.5, smosh(0.75, bananaT, 0.1));
    Hit result = Hit( d, 0., p, uv, transform, vec3(1.), 0., 0., 0., M_BANANA);
    vec3 bg = vec3(103., 202., 230.) / 255. * 0.5;
    vec3 xxpr = opTx(xxp, rotateZ(mix(0., -PI / 16., elongateT)));
    result = add(result, Hit(sdBox(xxpr - vec3(0., -10. * leaveT, 7.), vec3(10., -1. + (2.8 + 0.1 * xxp.x) * (elongateT), 1.)), 0., xxpr, xxpr, translate(vec3(0.)), Z(bg * 1.1 + 0.3), 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg * 1.1, 1., 0., 0., M_BG));
    return result;
}
#endif

Hit mapGrapeHalf(vec3 p, mat4 transform, float cutRotation, float amount) {
    float d = sdSphere(p, 1.); 
    vec3 cutp = p;
    //vec3 cutp = opTx(p, rotateY(cutRotation));
    Hit result = Hit(d, 0., p, p, transform, vec3(1.), 1., 0., 0., M_GRAPE);

    p += vec3(0., 0., 1.);
    vec3 topP = opTx(p, rotateX(PI / 2.));
    float r = (0.5 + (1. + floor(1.2 + p.z * 4.)) * 0.1) * 0.1;
    result = smadd(result, Hit(sdRoundedCylinder(topP, r, .05, .15), 0., p, p, transform, vec3(157., 148., 81.) / 255., 1., 0., .5, M_BG), 0.1);

    result.distance = opSmoothSubtraction(sdBox(cutp - vec3(2.02, 0., 0.), amount * vec3(2.)), result.distance, 0.03);
    return result;
}

#ifdef IS_GRAPE
Hit mapGrape(vec3 p) {
    vec3 np = p;
    vec3 xxp = np;

    float scale = .4;


    p /= scale;

    float t = mod(frame * 190. / 60. / 60. / 4. / 2., 1.);
    float rotation = t * 10.;

    float offset = floor(p.x / 3. + 0.5)  + 1.;

    float dropT1 = smosh(0.25 * offset + 0.1, t, 0.15);
    float offT = smosh(.82 + offset * 0.02, t, 0.15);

    float cutT = smosh(0.25 * offset + 0.215, t, 0.15);

    p.y += mix(-5., 0., dropT1);
    p.y += mix(0., 5.1, offT);
    

    p = opRepLim(p, 3., vec3(1., 0., 0.));

    rotation += offset;

    vec3 dropP = p;
    dropP.y += cutT * 8.;

    mat4 transform = translate(vec3(0.));
    transform *= rotateY(rotation);
    transform *= rotateX(-1.2);
    p = opTx(p, transform);
    dropP = opTx(dropP, transform);

    p.z *= 0.9;
    dropP.z *= 0.9;

    float cutRotation = t * 10. + offset * 4.;
    Hit result = mapGrapeHalf(p, transform, cutRotation, cutT);

    //result = add(result, mapGrapeHalf(dropP, transform, cutRotation, -1.));

    result.distance *= scale;

    float ender = smosh(2708., frame, 2727. - 2708.);
    vec3 bg = vec3(119., 74., 142.) / 255.;
    vec3 bg2 = vec3(62., 18., 85.) / 255.;
    vec3 bg3 = vec3(92., 17., 52.) / 255. * 1.5 + 0.3;
    vec3 bg4 = vec3(92., 17., 52.) / 255. * 1.1;
    bg = mix(bg, bg3, ender);
    bg2 = mix(bg2, bg4, ender);

    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 7.), vec3(10., 1.5, 1.)), 0., xxp, xxp, translate(vec3(0.)), bg, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg2, 1., 0., 0., M_BG));
    return result;
}
#endif


Hit mapGlass(vec3 p) {

    p = opTx(p, rotateX(sin(frame * 0.1) * 0.1 + PI / 8.));

    float d = sdSphere(p, 1.);

    d = max(d, -sdSphere(p, 0.95));

    d = opSmoothSubtraction(sdBox(p - vec3(0., 2., 0.), vec3(2.)), d, 0.02);
    
    mat4 transform = translate(vec3(0.));
    Hit result = Hit(d, 0., p, p, transform, vec3(1.), 0., 0., 0., M_GLASS);

    vec3 bg = vec3(223., 245., 191.) / 255.;

    result = add(result, Hit(
        sdBox(p + vec3(0., 0., -10.), vec3(10., 10., 1.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    result = add(result, Hit(
        sdBox(p + vec3(0., 0., 13.), vec3(10., 10., 1.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    result = add(result, Hit(
        sdBox(p + vec3(10., 0., 0.), vec3(1., 10., 10.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    result = add(result, Hit(
        sdBox(p + vec3(-10., 0., 0.), vec3(1., 10., 10.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    result = add(result, Hit(
        sdBox(p + vec3(0., -10., 0.), vec3(10., 1., 10.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    result = add(result, Hit(
        sdBox(p + vec3(0., 10., 0.), vec3(10., 1., 10.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    vec3 strawP = p + vec3(-0.2, 0.2, 0.);
    strawP = opTx(strawP, rotateZ(3. * PI / 8.));
    result = add(result, Hit(
        max(sdCappedCylinder(strawP, .045, 1.),
            -sdCappedCylinder(strawP, .04, 2.)),
        0., p, p, transform, bg, 0., 0., 0., M_BG));

    result = add(result, Hit(
        opSmoothSubtraction(sdBox(p - vec3(0., 1.7 + sin(p.x * 2. + frame * 0.1) * 0.1, 0.), vec3(2.)), sdSphere(p, .95), 0.1),
        0., p, p, transform, vec3(1., .6, 1.), .2, 0., 0., M_LIQUID_SMOOTHIE));
    return result;
}


#ifdef IS_KIWI
Hit mapKiwi(vec3 p) {

    vec3 xxp = vec3(p);

    float scale = 1.3;
    p *= scale;

    float t = mod(frame / 60. / 60. * 190. / 4. / 2., 1.);

    vec3 np = p;

    mat4 transform = translate(vec3(0., mix(-4., 0., smosh(0.13, t, 0.1)), 0.));


    float outT = smosh(0.9, t - p.x * 0.005, 0.1);
    float splitMove = mix(0., .6, smosh(0.20, t, 0.05));
    float rotateSplit = mix(0., 0.5, smosh(0.20, t, 0.05));
    float splitSpinner = 1.5 * pow(splitMove * 8., .5) * t;


    float moveToPeel = mix(0., 1., smosh(0.4, t, 0.1));

    float moveToSlicing = mix(0., 1., smosh(0.65, t, 0.1));

    float slicer = mix(0., 1., smosh(0.65, t, 0.1));

    float peelShrinker = mix(0., 0.70, smosh(0.55, t, 0.05));

    p.y += mix(0., 3., outT);



    mat4 leftT = transform;
    mat4 rightT = transform;
    leftT *= rotateZ(moveToPeel * PI / 2. + moveToSlicing * PI / 2.);
    leftT *= translate(vec3(splitMove - moveToPeel * 0.333 + moveToSlicing * 0.00, -moveToPeel * .5 + moveToSlicing * 0.5, 0.));
    leftT *= rotateY(-rotateSplit + moveToPeel * 0.333 - moveToSlicing * 0.333);
    leftT *= rotateX(splitSpinner);

    rightT *= rotateY(PI);
    rightT *= rotateZ(-moveToPeel * PI / 2. - moveToSlicing * PI / 2.);
    rightT *= translate(vec3(splitMove - moveToPeel * 0.333 + moveToSlicing * 0.00, +moveToPeel * .5 - moveToSlicing * 0.5, 0.));
    rightT *= rotateY(rotateSplit - moveToPeel * 0.333 - moveToSlicing * 0.333 * 2.);
    rightT *= rotateX(splitSpinner);


    Hit result = kiwihalf(p, leftT, t, slicer);


    result = add(result, kiwihalf(p, rightT, t, slicer));

    result.distance += peelShrinker * 0.1;
    result.distance /= scale;

    float transition = smosh(1496., frame, 1515. - 1496.);

    vec3 bg = vec3(226., 250., 192.) / 255. * 0.8;
    vec3 bg2 = vec3(110., 85., 55.) / 255. * 1.1;

    vec3 nextBg = vec3(103., 202., 230.) / 255. * 0.5 * 1.1;

    bg = mix(bg, nextBg, transition);
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 7.), vec3(mix(1.4, 5., transition), 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg, 1., 0., 0., M_BG));
    result = add(result, Hit(sdBox(xxp - vec3(0., 0., 9.), vec3(10., 10., 1.)), 0., xxp, xxp, translate(vec3(0.)), bg2, 1., 0., 0., M_BG));

    return result;

}
#endif


Hit map(vec3 p, float isShadowMap) {

    if(frame > 5110. - 0.5) {
#ifdef IS_BLENDER
        return mapBlender(p);
#endif
        //} else if(frame > 5098. - 0.5) {
} else if(frame > 5001. - 0.5) {


#ifdef IS_BANANA
} else if(frame > 4546. - 0.5) {
    return mapBanana(p);
#endif
} else if(frame > 3940. - 0.5) {
#ifdef IS_MANDARIN
    return mapMandarin(p, isShadowMap);
#endif
} else if(frame > 3334. - 0.5) {
#ifdef IS_STRAWBERRY
    return mapStrawberry(p);
#endif
} else if(frame > 2727. - 0.5) {
#ifdef IS_RASPBERRY
    return mapRaspberry(p);
#endif
} else if(frame > 2121. - 0.5) {
#ifdef IS_GRAPE
    return mapGrape(p);
#endif
#ifdef IS_PEACH
} else if(frame > 1515. - 0.5) {
    return mapPeach(p);
#endif
} else if(frame > 908. - 0.5) {
#ifdef IS_KIWI
    return mapKiwi(p);
#endif
} else {
    return mapGlass(p);
}
}




vec3 leafTexture(vec2 p) {
    float d = length(p.xy - vec2(0., 2.25));

    vec3 leafAlbedo = vec3(137., 163., 42.) / 255.;

    d = 0.1 * pow(sin((abs(p.x) - p.y + 0.005 * sin(abs(p.x) * 50. + p.x *11.321 + p.y * 3.23)) * 30.), 16.);
    d += 0.1 *(sin(p.x *.321) + cos(p.y *5.32) * sin(cos(p.y) * sin(p.x * 43.2)));
    return leafAlbedo + d;
}


Hit march(vec3 rayOrigin, vec3 rayDirection, float side) {
    float distance = 0.;
    float maxDistance = 50.;
    Hit hit;
    for(int i = 0; i < 256; i++) {
        vec3 position = rayOrigin + rayDirection * distance;
        hit = map(position, 0.);
        hit.distance *= side;
        hit.steps = float(i);
        hit.position = position;
        if(hit.distance < EPSILON || distance > maxDistance) {
            break;   
        }
        distance += hit.distance * 0.9;
    }
    hit.distance = distance;
    return hit;
}

float softshadow( vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float ph = 1e20;
    float t = mint;
    for( int i = 0; i < 128; i++)
    {
        if(t >= maxt){
            break;
        }
        float h = map(ro + rd*t, 1.).distance *0.9;
        if( h<0.001 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h;
    }
    return res;
}


vec3 calculateNormal(vec3 p) {
    return normalize(vec3(
                map(vec3(p.x + EPSILON, p.y, p.z), 0.).distance - map(vec3(p.x - EPSILON, p.y, p.z), 0.).distance,
                map(vec3(p.x, p.y + EPSILON, p.z), 0.).distance - map(vec3(p.x, p.y - EPSILON, p.z), 0.).distance,
                map(vec3(p.x, p.y, p.z  + EPSILON), 0.).distance - map(vec3(p.x, p.y, p.z - EPSILON), 0.).distance
                ));
}


#ifdef IS_KIWI
vec4 kiwiTexture(vec3 uv) {

    uv = opTx(uv, rotateY(PI / 2.));

    uv /= 0.99;

    //uv.x *= .5;


    float zOffset = pow(1. - abs(uv.z), .5);

    float angle = atan(uv.y, uv.x);


    float lenScale = 1.25;

    float whiteLimit = 0.15;
    whiteLimit *= 1. + 0.2 * sin(uv.x * 8.) * sin(angle * 3.) + 0.1 * cos(angle * 3.);

    float len = length(uv);
    len = pow(len, 2.) * lenScale;

    //len += 1. - zOffset;


    if(length(uv) > 0.999) {
        float f = 0.1 * (1. - bubble(20. * vec3(angle * 5., uv.y, 0.)));
        vec2 s = vec2(0.001, 0.);
        vec3 eX = s.xyy;
        vec3 eY = s.yxy;
        vec3 eZ = s.yyx;
        uv *= 18.;
        vec2 res = voronoi3(uv, 1.);

        float j = (res.x - res.y * 0.75) * 20.;

        res = voronoi3(100. + uv, 1.);

        j = smosh(0.30, max(0., j), 0.4);

        uv *= 1.5;
        j = mix(j, (res.x - res.y * 0.75) * 20., 0.5);
        j = smosh(0.6, max(0., j), 0.1);

        f *= 0.5;
        j *= 0.08;

        vec3 color = vec3(157., 131., 55.) / 255. + f + j; 

        color = pow(color, vec3(1.1));

        return vec4(color , f + j * 2.);  
    }


    vec3 outerGreen = vec3(65., 145., 40.) / 255.;
    vec3 innerGreen = vec3(182., 201., 115.) / 255.;
    innerGreen = vec3(143., 204., 60.) / 255.;
    innerGreen = vec3(185., 219., 106.) / 255.;

    vec3 darkTapGreen = vec3(64., 114., 11.) / 255. * 1.2;
    vec3 whiteCenter = vec3(251., 251., 198.) / 255.;
    vec3 whiteCenterBorderish = vec3(122., 133., 29.) / 255.;
    vec3 whiteCenterBorder = vec3(170., 181., 78.) / 255.;

    float taps = 36. / 2.;

    angle += 0.02 * sin(angle * taps * 2.);
    float normalizedLen = length(uv.xy);


    vec3 color = mix(innerGreen, outerGreen, pow(Z((normalizedLen - whiteLimit * lenScale) / (1. - whiteLimit * lenScale)), 3.));

    color = mix(color, darkTapGreen, 0.8 * max(step(len, whiteLimit), smosh(len - whiteLimit + 0.4, abs(sin(angle * taps)), .5 )));

    color = mix(color, darkTapGreen, 0.1 * max(step(len, whiteLimit), pow(smosh(len - whiteLimit + 0.3, abs(pow(sin(angle * taps), 32.)), .1 ), 32.)));



    color = mix(color, whiteCenterBorder, smosh(len, whiteLimit + .1, .1));

    color = mix(color, whiteCenterBorderish, smosh(len, whiteLimit + .02, .02));

    color = mix(color, whiteCenter, smosh(len, whiteLimit, 0.01));

    color = mix(color, whiteCenter, max(step(len, whiteLimit), smosh(len - whiteLimit + 0.85, abs(pow(sin(PI / 2. + angle * taps), .15)), 0.15)));
    color = mix(color, whiteCenter, max(step(len, whiteLimit), smosh(len - whiteLimit + .88, abs(pow(sin(angle * taps), 32.)), .15)));


    float seedAmount = pow(
            Z(2. * (1. - smosh(0.3, len, 0.2)) * smosh(whiteLimit, len , .05) * abs(sin(angle * taps)) * abs(cos(sin(angle * 5.) + sin(angle * 3.) + sin(angle * 7.) + sin(angle * 11.) + pow(len, .5) *20.)) * .5),
            4.);

    color = mix(color, vec3(0.), seedAmount);

    color = pow(color, vec3(1.2));

    return vec4(color, seedAmount);   
}
#endif

#ifdef IS_MANDARIN
MaterialProperties mandarinPeelTexture(vec3 uv) {
    vec2 res = voronoi3(uv * 20., 1.);

    vec3 outerColor1 = vec3(243., 159., 27.) / 255.;
    vec3 outerColor2 = vec3(246., 160., 16.) / 255.;
    vec3 outerColor = mix(outerColor1, outerColor2, snoise(uv));
    vec3 innerColor = vec3(241., 225., 214.) / 255.;
    float len = length(uv);
    vec3 color = mix(innerColor, outerColor, smosh(0.97, len, 0.01));

    float bump = 0.5;
    float micro = snoise(uv * 32.) * snoise(uv * 2.);
    float roughness = 0.2 +0.3 * micro;
    float dimples = pow(Z( 3. * ((1. - res.x) - .6)), 4.);

    bump = 1. -dimples * 0.5 + micro * 0.05;

    bump *= 0.6;
    
    return MaterialProperties(color, roughness,bump);
}
#endif

#ifdef IS_PEACH
MaterialProperties peachSeedTexture(vec3 uv) {
    float roughness = 0.5 + 0.2 * snoise(uv * 16.);
    float bump = 0.5 + 0.1* snoise(uv * 32.);
    vec3 color = vec3(171., 113., 65.) / 255.;
    float len = length(uv);
    color *= 1. - 0.25 * (1. - smosh(0.3, len, 0.02));
    return MaterialProperties(color, roughness, bump);
}
#endif

#ifdef IS_PEACH
MaterialProperties peachTexture(vec3 uv) {
    uv *= 2.;
    float bump = 0.5;
    float roughness = 1.;
    vec3 color1 = vec3(252., 198., 66.) / 255.;
    vec3 color2 = vec3(242., 63., 69.) / 255.;
    vec3 color3 = vec3(172., 26., 71.) / 255.;
    float mixer = snoise(uv * .5);
    vec3 color = mix(color1, color2, Z(mixer * 2.));
    color = mix(color, color3, Z(mixer * 2. - 1.));
    color = mix(color, vec3(1), 0.25 * (snoise(uv * 64.) + snoise(uv * .5)));

    float len = length(uv);
    float skinMixer = pow(1. - smosh(0.5, len, 0.2), 0.45);

    vec3 meatYellow = vec3(255., 206., 67.) / 255.;
    vec3 meatRed = vec3(184., 28., 5.) / 255.;

    float cells = pow(voronoi3(uv * 30., 1.).x, .25);
    meatYellow *= .95 + 0.1 * cells;
    bump += cells * 0.1;
    roughness *= 1. - 0.9 * skinMixer;
    roughness = mix(roughness + cells * 0.5, roughness, skinMixer);

    float angle = atan(uv.z, uv.y);
    vec3 insideColor = mix(meatRed, meatYellow, smosh(0.21, len + sin(angle * 8.) * len * 0.02 * sin(angle * 64.) + 0.01 * snoise(uv * 32.), 0.15));

    color = mix(color, insideColor, skinMixer);


    return MaterialProperties(color, roughness, bump);
}
#endif

#ifdef IS_BANANA
MaterialProperties bananaTexture(vec3 uv) {
    float bump = 0.5;
    float roughness = 0.9;
    vec3 yellow = vec3(242., 223., 108.) / 255.;
    vec3 green = vec3(162., 194., 2.) / 255.;
    vec3 brown = vec3(83., 46., 25.) / 255.;

    vec3 lightInside = vec3(251., 228., 174.) / 255.;
    vec3 darkInside = vec3(202., 138., 43.) / 255.;


    vec3 color = mix(green, yellow,  smosh(0., uv.z, 0.2));
    color = mix(color, brown, smosh(0.98, uv.z, 0.015));

    float len = length(uv.xy);

    float angle = atan(uv.y, uv.x);

    vec2 res = voronoi3(uv * 4., 1.);

    float insideMixer = 0.5 + 0.5 * sin(angle * 3.) + 0.15 * sin(angle * 16.);
    insideMixer *= (1. - len);
    insideMixer = mix(0., insideMixer, smosh(0., len, 0.05));
    //insideMixer = smosh(0.25, insideMixer, 0.5);
    insideMixer = mix(insideMixer, insideMixer * res.x, 0.5);
    insideMixer = pow(insideMixer, 2.) * 2.;
    insideMixer = Z(insideMixer);
    vec3 insideColor = mix(lightInside, darkInside, insideMixer);

    float skinMixer = smosh(1.5, len, 0.2);
    color = mix(insideColor, color, skinMixer);
    color += 0.05 * smosh(1.2, len, 0.3) * (1. - skinMixer);

    float noiz = snoise(uv * 0.125) + snoise(uv * 0.25) + snoise(uv * 0.5) + snoise(uv) + snoise(uv * 2.) + snoise(uv * 4.) + snoise(uv * 8.);
    noiz /= 8.;

    color *= (1. - noiz * .01);

    float bananaEdgePoint = 1.4;
    float bananaEdgeMixture =(1. - smosh(bananaEdgePoint + 0.3, len, 0.01));

    float stripes = max(pow(noiz, 0.25), 0.6);

    float edgeBump =  pow(0.1 * snoise(uv * vec3(16., 16., 1024.)) + max(stripes, 0.6 + 0.05 * sin(uv.z * 800.)), 2.) + 0.25 * smosh(0.61, stripes, 0.1);
    color += 0.05 * smosh(0.61, stripes, 0.01) * bananaEdgeMixture * (1. - skinMixer);

    color = mix(color, vec3(1.), 0.2 * bananaEdgeMixture);

    float peelMixer = smosh(0.34, uv.z, 0.00001) * (1. - smosh(.73, uv.z, 0.00001));

    bump = mix(2. * bump, edgeBump * len / 2., peelMixer * bananaEdgeMixture);
    //bump = mix(bump, insideMixer, 0.5);

    //bump = 0.5 + bump * 0.1;

    roughness = mix(0.3, 0.9, skinMixer);

    return MaterialProperties(color, roughness, bump);
}
#endif

#ifdef IS_MANDARIN
MaterialProperties mandarinTexture(vec3 uv) {
        float len = length(uv);
        vec3 color = vec3(247., 181., 32.) / 255.;
        vec2 res = voronoi3(uv * vec3(2.5, 1., 1.) * 4.5 ,1.);
        float x = pow(res.x, 2.);
        float bump = pow(1. - x, 2.);
        x = mix(x, pow(x, 0.5), smosh(1.35, len, 0.1));
        x = mix(x, pow(x, 0.1), 1. - smosh(0.01, abs(uv.z), 0.01));
        x = Z(x);
        color = mix(color, vec3(1.), vec3(x) * 0.5);
        color = pow(color, vec3(1.2));
        return MaterialProperties(color, 0.05 + x * 0.5, bump);
}
#endif

#ifdef IS_RASPBERRY
MaterialProperties raspberryTexture(vec3 uv) {
    vec3 color = vec3(249., 27., 70.) / 255.;
    float roughness = 0.;
    for(int i = 1; i < 5; i++) {
        roughness += snoise(uv * float(i * i)) / float(i);
    }
    roughness *= 0.5;
    float bump = roughness;
    color = mix(color, vec3(1.), roughness * 0.2);

    color = mix(color * 0.5, color, smosh(0.75, length(uv), 0.2));
    roughness = 0.2 + 0.3 * roughness;
    roughness = mix(.8, roughness, smosh(0.75, length(uv), 0.2));
    return MaterialProperties(color, roughness, 0.5 + bump); 
}
#endif

#ifdef IS_STRAWBERRY
MaterialProperties strawberryTexture(vec3 uv) {
    vec3 color = vec3(222., 71., 65.) / 255.;


    float fuzz = sin(73.42 * sin(uv.x * 23.341) * uv.x) * cos(uv.y *23.123) * cos(sin(uv.z *321.234)) * 0.1;

    float ringlen = length(uv.zx);

    ringlen -= 0.2 * (0.5 + 0.5 * sin(1. + uv.y * 3.));

    float amount = pow((0.5 + 0.5 * abs(sin(ringlen * 10.))), 4.) * (1. - max(0., pow(ringlen, .5)));

    amount += fuzz;

    float m = 0.1 + snoise(uv * 0.5) * (snoise(uv * 8.) / 2. + snoise(uv * 16.) / 3. + snoise(uv * 32.) / 4. + snoise(uv * 64.) / 5.);

    color = mix(color, vec3(1.), amount * 0.5 + m);

    color = pow(color, vec3(2.));

    return MaterialProperties(
            color,
            m,
            amount); 
}
#endif


#ifdef IS_GRAPE
MaterialProperties grapeTexture(vec3 uv) {

    float len = length(uv);

    vec3 baseColor = vec3(222., 232., 134.) / 255.;

    vec3 lightColor = vec3(233., 233., 194.) / 255.;
    vec3 seedColor = vec3(123., 103., 41.) / 255.;

    float ringlen = length(uv.xy);

    vec3 lightGrapeSkin = vec3(191., 194., 34.) / 255.;
    vec3 darkGrapeSkin = vec3(178., 142., 43.) / 255.;

    float angle = atan(uv.y, uv.x);


    float amount = 0.5 + 0.5 * pow(sin(ringlen * 16. * (pow(len, 2.) - .42)), 8.);

    amount *= 1. - smosh(0.9, len, 0.09);

    float seeds = Z(pow(max(amount * (1. - smosh(0.5, len , 0.1)) - 0.5, 0.), 4.) * 8.);

    seeds *= max(0., sin(PI / 2. + angle * 4.) * sin(PI / 2. * 3. + angle * 2.));
    amount *= 0.5;
    vec3 color = mix(baseColor, lightColor, amount);

    float lines = 1. - pow(sin(1000. + ringlen * 40. * len), 16.) * (1. - len) * 0.15;

    color *= lines;




    float skinColorAmount = smosh(0.99, len, 0.01);
    float noiz = snoise(uv * 2. * vec3(40., 1., 1.));
    float skinColorModulator = (0.8 + 0.2 * noiz) * (0.5 + 0.5 * sin(2. + uv.z * 2.));
    vec3 skinColor = mix(lightGrapeSkin, darkGrapeSkin, skinColorModulator);

    float noiz2 = snoise(uv * 5.);


    color = mix(color, skinColor, skinColorAmount);

    color = mix(color, seedColor, seeds);

    if(len > 1.01) {
        vec3 bg = vec3(223., 245., 191.) / 255.;
        color = bg;
    }



    return MaterialProperties(color, mix(0.5, 0.4 * skinColorModulator + 0.1 + 0.5 *noiz2 * noiz, smosh(0.9, len, 0.1)), 0.5 - amount * 0.2 + seeds + lines * 0.1);
}
#endif



vec3 image(vec2 uv) {


    vec2 iResolution = vec2(resolutionX, resolutionY);

    uv.x *= iResolution.x / iResolution.y;

    vec3 bg = vec3(223., 245., 191.) / 255.;

    if(frame > 3334. - 0.5) {
        bg = vec3(230., 227., 204.) / 255. * mix(1., 0.8, smosh(1. / 3. * 0.9, abs(uv.x), 0.001));
    } else if(frame > 2727. - 0.5) {
        bg = vec3(92., 17., 52.) / 255. * mix(1.1, 1.5, smosh(1. / 3. * (1. - smosh(3296., frame, 3315. - 3296.)), abs(uv.y), 0.001));
    }
    

    float splitIndex = 0.;
    if(frame > 5000. && frame < 5100.) {
        /* TODO the +.15 is fudged */
        float width = iResolution.x / iResolution.y / 6.;
        splitIndex = floor(uv.x / width);
        uv.x = repeat(uv.x + 0.15, iResolution.x / iResolution.y / 6.);

        uv *= 2.;

        if(splitIndex < -2.5) {
#ifdef IS_KIWI
            return kiwiTexture(vec3(uv, 0.)).rgb;
#endif
        } else if(splitIndex < -1.5) {
#ifdef IS_KIWI
            return kiwiTexture(vec3(uv, 0.)).rgb;
#endif
        } else if(splitIndex < -0.5) {
#ifdef IS_KIWI
            return kiwiTexture(vec3(uv, 0.)).rgb;
#endif
        } else if(splitIndex < 0.5) {
#ifdef IS_GRAPE
            return grapeTexture(vec3(uv, 0.)).albedo;
#endif
        } else if(splitIndex < 1.5) {
#ifdef IS_MANDARIN
            return mandarinTexture(vec3(uv, 0.)).albedo;
#endif
        } else if(splitIndex < 2.5) {
#ifdef IS_STRAWBERRY
            return strawberryTexture(vec3(uv, 0.)).albedo;
#endif
        }
        /*
       return mapGrape(p);
       } else if(frame > 5077. - 0.5) {
       return mapStrawberry(p);
       } else if(frame > 5055. - 0.5) {
       return mapGrape(p);
       } else if(frame > 5033. - 0.5) {
       return mapStrawberry(p);
       } else if(frame > 5023. - 0.5) {
       return mapKiwi(p);
       } else if(frame > 5001. - 0.5) {
       return mapGrape(p);
       */
    }




    /*
       float frameDist = length(max(abs(uv) - 0.85, 0.)) - .15;
       if(frameDist > 0.) {


       float drop = length(max(abs(uv + vec2(0., 0.025)) - 0.85, 0.)) - .15;
       return vec3(59., 77., 34.) / 255.* 0.8 * (2. / 3. + 1. / 3. * smosh(0., drop, 0.025));
       }
     */



    float snare = pow(max(0., 1. - mod(frame * 190. / 60. / 60. / 2. + 1., 2.)), 2.) * 2.;

    vec3 cameraPosition = vec3(0., 0., -12.);
    vec3 rayDirection = normalize(vec3(uv, 4. - 0.2 * snare));

    if(frame > 5153. - 0.5) {
        cameraPosition = vec3(.6, 3., -9.);
        rayDirection = opTx(rayDirection, rotateZ(0.1 - (frame - 5153.) * 0.0002));
        rayDirection = opTx(rayDirection, rotateX(0.22));
        rayDirection.x += (frame - 5153.) * 0.0001;
    } else if(frame > 5110. - 0.5) {
        vec3 cameraPosition = vec3(0., 1.3, -5.);
    }

    cameraPosition.z += snare;

    if(frame >  5153. - 0.5) {
        /*
        float shakespeed = smosh(5153., frame, 7578. - 5153.) * 0.05;
        float shakeamount = smosh(5153., frame, 7578. - 5153.) * 0.01 + 0.001;
        float noiz = snoise(vec3(frame * shakespeed, 0., 0.)) - 0.5;
        float noiz2 = snoise(vec3(frame * shakespeed, 10., 10.)) - 0.5;
        float noiz3 = snoise(vec3(frame * shakespeed, 50., 50.)) - 0.5;
        rayDirection = opTx(rayDirection, rotateX(noiz * shakeamount));
        rayDirection = opTx(rayDirection, rotateY(noiz2 * shakeamount));
        rayDirection = opTx(rayDirection, rotateZ(noiz3 * shakeamount));
        */
    }


    Hit hit = march(cameraPosition, rayDirection , 1.);


    if(length(hit.position - cameraPosition) > 50.) {
        return bg;
    }


    vec3 normal = calculateNormal(hit.position);



    vec3 light1Position = vec3(-20., 10., 10.);
    vec3 light1Direction = normalize(light1Position - hit.position);

    vec3 light2Position = cameraPosition + vec3(2., 0., 0.);
    vec3 light2Direction = normalize(light2Position - hit.position);
    float bump = 0.5;
    vec3 bumpNormal = vec3(0.);

    float bubbleAmount = 0.;
    float bubbleScale = 8.;
    float fakeSSSAmount = 1.;

    mat4 noTranslate = hit.transform;
    noTranslate[0][3] = 0.;
    noTranslate[1][3] = 0.;
    noTranslate[2][3] = 0.;
    noTranslate /= noTranslate[3][3];
    float ep = 0.02;
    vec3 eX = opTx(vec3(ep, 0., 0.), noTranslate);
    vec3 eY = opTx(vec3(0., ep, 0.), noTranslate);
    vec3 eZ = opTx(vec3(0., 0., ep), noTranslate);

    float backsideSide = -1.;
    float backsideRayDirectionStep = 0.01;
    if(CHECK_MATERIAL(hit.material, M_BLENDER)) {
#ifdef IS_MANDARIN
    } else if(CHECK_MATERIAL(hit.material, M_MANDARIN_PEEL)) {
        MaterialProperties mp = mandarinPeelTexture(hit.uv);
        hit.albedo = mp.albedo;
        hit.emissive = 0.3;
        hit.roughness = mp.roughness;
        bubbleAmount = 0.;
        fakeSSSAmount = 0.;
        backsideSide = -1.;
        backsideRayDirectionStep = .5;

        bumpNormal = -vec3(
                mandarinPeelTexture(hit.uv + eX).bump - mandarinPeelTexture(hit.uv - eX).bump,
                mandarinPeelTexture(hit.uv + eY).bump - mandarinPeelTexture(hit.uv - eY).bump,
                mandarinPeelTexture(hit.uv + eZ).bump - mandarinPeelTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal);
#endif
#ifdef IS_MANDARIN
    } else if(CHECK_MATERIAL(hit.material, M_MANDARIN)) {
        MaterialProperties mp = mandarinTexture(hit.uv);
        hit.albedo = mp.albedo;
        hit.roughness = mp.roughness;
        bubbleAmount = 1.;
        bubbleScale = 1.;
        fakeSSSAmount = 1.;
        backsideSide = -1.;
        backsideRayDirectionStep = .5;

        bumpNormal = -vec3(
                mandarinTexture(hit.uv + eX).bump - mandarinTexture(hit.uv - eX).bump,
                mandarinTexture(hit.uv + eY).bump - mandarinTexture(hit.uv - eY).bump,
                mandarinTexture(hit.uv + eZ).bump - mandarinTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal);
        fakeSSSAmount *= (1. - hit.roughness);
        fakeSSSAmount *= 2.;
#endif

#ifdef IS_BLENDER
    } else if(CHECK_MATERIAL(hit.material, M_GLASS)) {

        float refractiveIndex = 1. / 1.333;

        vec3 refracted1 = refract(rayDirection, normal, refractiveIndex);
        vec3 reflected1 = reflect(rayDirection, normal);
        vec3 raydir = refracted1;

        float inside = 1.;
        
        vec3 albedo = vec3(1.);

        if(length(refracted1) < 0.1) {
            raydir = normalize(reflected1);
        } else {
            raydir = normalize(refracted1);
            inside = -inside;
        }

        float inch = 0.001;
        Hit newHit = march(hit.position + raydir * inch, raydir, inside);

        albedo = vec3(length(hit.position - newHit.position));

        float angle = atan(hit.uv.z, hit.uv.x);
        float whitenesser = smosh(.6 + 0.05 * sin(angle), hit.uv.y, 0.02);
        float whiteness = mix(0.25, 0.2, whitenesser);
        vec3 whiteTint = mix(vec3(195., 128., 224.) / 255., vec3(1.), 0.8 + 0.2 * whitenesser);

        if(CHECK_MATERIAL(newHit.material, M_GLASS)) {
            vec3 norm = calculateNormal(newHit.position) * inside;
            vec3 refracted2 = refract(raydir, norm, inside > 0. ? 1. / 1.333 : 1.333);
            vec3 reflected2 = reflect(raydir, norm);
            if(length(refracted2) < 0.1) {
                raydir = normalize(reflected2);
            } else {
                raydir = normalize(refracted2);
                inside = -inside;
            }
            newHit = march(newHit.position + raydir * inch, raydir, inside);

            if(CHECK_MATERIAL(newHit.material, M_GLASS)) {
                norm = calculateNormal(newHit.position) * inside;
                vec3 refracted3 = refract(raydir, norm, inside > 0. ? 1. / 1.333 : 1.333);
                vec3 reflected3 = reflect(raydir, norm);
                if(length(refracted3) < 0.1) {
                    raydir = normalize(reflected3);
                } else {
                    raydir = normalize(refracted3);
                    inside = -inside;
                }
                newHit = march(newHit.position + raydir * inch, raydir, inside);

                if(CHECK_MATERIAL(newHit.material, M_GLASS)) {
                    norm = calculateNormal(newHit.position) * inside;
                    vec3 refracted4 = refract(raydir, norm, inside > 0. ? 1. / 1.333 : 1.333);
                    vec3 reflected4 = reflect(raydir, norm);
                    if(length(refracted4) < 0.1) {
                        raydir = normalize(reflected4);
                    } else {
                        raydir = normalize(refracted4);
                        inside = -inside;
                    }
                    newHit = march(newHit.position + raydir * inch, raydir, inside);

                    albedo = newHit.albedo * fancyLighting(newHit, light2Direction, norm, -raydir, 1.);
                } else {
                    albedo = newHit.albedo * fancyLighting(newHit, light2Direction, norm, -raydir, 1.);
                }

                albedo = whiteTint * whiteness + albedo * (1. - whiteness);
            } else {
                albedo = newHit.albedo * fancyLighting(newHit, light2Direction, norm, -raydir, 1.);
            }
            albedo = whiteTint * whiteness + albedo * (1. - whiteness);
        } else {
            vec3 norm = calculateNormal(newHit.position) * inside;
            albedo = newHit.albedo * fancyLighting(newHit, light2Direction, norm, -raydir, 1.);
        }

        albedo = whiteTint * whiteness + albedo * (1. - whiteness);


        hit.albedo = albedo;
        fakeSSSAmount = 1.;
#endif

#ifdef IS_BLENDER
    } else if(CHECK_MATERIAL(hit.material, M_BLENDER_GLASS)) {

        vec3 refracted = refract(rayDirection, normal, 1.52);
        vec3 reflected = reflect(rayDirection, normal);

        float split = 0.5;
        hit.albedo = split * vec3(1.) + (1. - split) * skybox(reflected).rgb;
        fakeSSSAmount = 3.;
#endif

#ifdef IS_BLENDER
    } else if(CHECK_MATERIAL(hit.material, M_BLENDER_GLASS_SMOOTHIE)) {

        vec3 refracted = refract(rayDirection, normal, 1.52);
        vec3 reflected = reflect(rayDirection, normal);

        vec3 purple = vec3(227., 152., 179.) / 255.;
        vec3 yellow = vec3(243., 212., 144.) / 255.;

        float angle = atan(hit.uv.z, hit.uv.x);


        float t = frame / 60.;
        vec3 smoothie = vec3(0.);
        for(int i = 0; i < 3; i++) {
            t += float(i) * 0.1;
            smoothie += 1. / 3. * mix(purple, yellow, .5 + 0.5 * sin(t * 17. + angle * (3. + 2. * sin(hit.uv.y + 1. * t)) + hit.uv.y * (1. + 0.1 * sin(t) * 50.) + sin(t * 11. + hit.uv.y * 100.)));
        }

        float split = 0.5;
        hit.albedo = split * smoothie + (1. - split) * skybox(reflected).rgb;
        fakeSSSAmount = 3.;
#endif

#ifdef IS_RASPBERRY
    } else if(CHECK_MATERIAL(hit.material, M_RASPBERRY)) {
        bubbleAmount = 0.;
        bubbleScale = 5.;
        MaterialProperties center = raspberryTexture(hit.uv);
        fakeSSSAmount = 1. + 10. * hit.metalness;
        fakeSSSAmount *= 0.5;
        hit.metalness = 0.;
        hit.albedo = center.albedo * 0.9;
        hit.roughness = center.roughness;
        bump = center.bump;
        bumpNormal = -vec3(
                raspberryTexture(hit.uv + eX).bump - raspberryTexture(hit.uv - eX).bump,
                raspberryTexture(hit.uv + eY).bump - raspberryTexture(hit.uv - eY).bump,
                raspberryTexture(hit.uv + eZ).bump - raspberryTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal * bubbleAmount);
#endif

#ifdef IS_PEACH
    } else if(CHECK_MATERIAL(hit.material, M_PEACH_SEED)) {
        bubbleAmount = .0;
        bubbleScale = 1.;
        MaterialProperties center = peachSeedTexture(hit.uv);
        fakeSSSAmount = 0.;
        hit.albedo = center.albedo;
        hit.roughness = center.roughness;
        bump = center.bump;
        bumpNormal = -vec3(
                peachSeedTexture(hit.uv + eX).bump - peachSeedTexture(hit.uv - eX).bump,
                peachSeedTexture(hit.uv + eY).bump - peachSeedTexture(hit.uv - eY).bump,
                peachSeedTexture(hit.uv + eZ).bump - peachSeedTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal);
#endif

#ifdef IS_PEACH
    } else if(CHECK_MATERIAL(hit.material, M_PEACH)) {
        bubbleScale = 1.2;
        MaterialProperties center = peachTexture(hit.uv);
        bubbleAmount = (1. - smosh(0.8, center.roughness, 0.01) ) * 0.5;
        fakeSSSAmount = 1.;
        hit.albedo = center.albedo;
        hit.roughness = center.roughness;
        bump = center.bump;
        bumpNormal = -vec3(
                peachTexture(hit.uv + eX).bump - peachTexture(hit.uv - eX).bump,
                peachTexture(hit.uv + eY).bump - peachTexture(hit.uv - eY).bump,
                peachTexture(hit.uv + eZ).bump - peachTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal);
#endif

#ifdef IS_BANANA
    } else if(CHECK_MATERIAL(hit.material, M_BANANA)) {
        bubbleAmount = .5;
        bubbleScale = 1.;
        MaterialProperties center = bananaTexture(hit.uv);
        fakeSSSAmount = .2;
        hit.albedo = center.albedo;
        hit.roughness = center.roughness;
        bump = center.bump;
        bumpNormal = -vec3(
                bananaTexture(hit.uv + eX).bump - bananaTexture(hit.uv - eX).bump,
                bananaTexture(hit.uv + eY).bump - bananaTexture(hit.uv - eY).bump,
                bananaTexture(hit.uv + eZ).bump - bananaTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal);
#endif

#ifdef IS_STRAWBERRY
    } else if(CHECK_MATERIAL(hit.material, M_STRAWBERRY)) {
        bubbleAmount = max(1. - smosh(0.9, length(hit.uv), 0.1) * 1.1, 0.);
        bubbleScale = 4.;
        fakeSSSAmount = 1.;
        MaterialProperties center = strawberryTexture(hit.uv);
        hit.albedo = mix(hit.albedo, center.albedo, Z(bubbleAmount));
        hit.roughness = center.roughness;
        bump = mix(0.5, center.bump, Z(bubbleAmount));
        bumpNormal = -vec3(
                strawberryTexture(hit.uv + eX).bump - strawberryTexture(hit.uv - eX).bump,
                strawberryTexture(hit.uv + eY).bump - strawberryTexture(hit.uv - eY).bump,
                strawberryTexture(hit.uv + eZ).bump - strawberryTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal * bubbleAmount);
#endif
#ifdef IS_GRAPE
    } else if(CHECK_MATERIAL(hit.material, M_GRAPE)) {


        MaterialProperties mp = grapeTexture(hit.uv);
        hit.albedo =  mp.albedo;
        hit.roughness = mp.roughness;
        bubbleAmount = 4.;
        bubbleScale = .5 + .2 * (sin(hit.uv.y * 2.321) + 0.4 * sin(hit.uv.z * 5.234));
        bubbleScale *= 0.5;
        fakeSSSAmount = .5;
        MaterialProperties center = grapeTexture(hit.uv);
        bump = center.bump;
        bumpNormal = -vec3(
                grapeTexture(hit.uv + eX).bump - grapeTexture(hit.uv - eX).bump,
                grapeTexture(hit.uv + eY).bump - grapeTexture(hit.uv - eY).bump,
                grapeTexture(hit.uv + eZ).bump - grapeTexture(hit.uv - eZ).bump);
        normal = normalize(normal + bumpNormal);

        bubbleAmount = .5;
#endif

#ifdef IS_KIWI
    } else if(CHECK_MATERIAL(hit.material, M_KIWI)) {
        backsideRayDirectionStep = .01;
        backsideSide = -1.;
        fakeSSSAmount = 1.5;
        hit.roughness = 0.5;
        vec3 q = hit.uv * 7.;
        vec4 center = kiwiTexture(hit.uv);
        hit.albedo = center.rgb;
        bump = center.a;
        bumpNormal = -vec3(
                kiwiTexture(hit.uv + eX).a - kiwiTexture(hit.uv - eX).a,
                kiwiTexture(hit.uv + eY).a - kiwiTexture(hit.uv - eY).a,
                kiwiTexture(hit.uv + eZ).a - kiwiTexture(hit.uv - eZ).a);
        normal = normalize(normal + bumpNormal);
        hit.roughness = max(0.5, hit.roughness - pow(bump, 0.5));
        if(length(hit.uv) > 0.99) {
            hit.roughness = 1.;   
        }
        bubbleAmount = 1. - smosh(0.8, length(hit.uv), 0.12);
#endif
    }

    if(hit.metalness > 0.1) {
        vec3 reflected = reflect(rayDirection, normal);
        hit.albedo += (1. - hit.roughness) * skybox(reflected).rgb;
    }

    /* bubbles */
    if(bubbleAmount > 0.) {
        vec3 q = hit.uv * 7.;
        float bubbleHeight = 0.1 * Z(bubble(q * bubbleScale)) * bubbleAmount;
        float height = Z(bubbleHeight * (1. - bump));
        vec3 bubbleNormal = -vec3(
                bubble(bubbleScale * (q + eX)) - bubble(bubbleScale * (q - eX)),
                bubble(bubbleScale * (q + eY)) - bubble(bubbleScale * (q - eY)),
                bubble(bubbleScale * (q + eZ)) - bubble(bubbleScale * (q - eZ))) * 2.;

        float bubbleNormalAmount = 1. - smosh(0.95, length(hit.uv), 0.05);

        bubbleNormal = mix(normal, bubbleNormal, bubbleNormalAmount);

        vec3 refracted = mix(rayDirection, refract(rayDirection, normalize(normal + bubbleNormal), 1.333), bubbleNormalAmount);
        vec3 reflected = reflect(rayDirection, normalize(normal + bubbleNormal));

        Hit newHit = march(hit.position - rayDirection * height, refracted , 1.); 
        if(false) {  // requried for when all cases are skipped
#ifdef IS_RASPBERRY
        } else if(CHECK_MATERIAL(hit.material, M_RASPBERRY)) {
            hit.albedo = raspberryTexture(newHit.uv).albedo + 0.1 * skybox(reflected).rgb + height * 0.1;
#endif
#ifdef IS_STRAWBERRY
        } else if(CHECK_MATERIAL(hit.material, M_STRAWBERRY)) {
            hit.albedo = strawberryTexture(newHit.uv).albedo + 0.1 * skybox(reflected).rgb + height * 0.1;
#endif
#ifdef IS_MANDARIN
        } else if(CHECK_MATERIAL(hit.material, M_MANDARIN)) {
            hit.albedo = mandarinTexture(newHit.uv).albedo + 0.1 * skybox(reflected).rgb + height * 0.1;
#endif
#ifdef IS_GRAPE
        } else if(CHECK_MATERIAL(hit.material, M_GRAPE)) {
            hit.albedo = grapeTexture(newHit.uv).albedo;
            vec3 uvDirection = opTx(rayDirection, newHit.transform);
            vec3 translucency = vec3(0.);
            for(float i = 1.0; i > -.0001; i -= 0.01) {
                translucency += grapeTexture(newHit.uv + uvDirection * i).albedo / 100.;
            }
            translucency = pow(translucency, vec3(4.));
            hit.albedo = mix(hit.albedo, translucency, 0.75);
            hit.albedo += 0.3 * skybox(reflected).rgb + height * 0.2;
#endif
#ifdef IS_BANANA
        } else if(CHECK_MATERIAL(hit.material, M_BANANA)) {
            hit.albedo = bananaTexture(newHit.uv).albedo + 0.1 * skybox(reflected).rgb + height * 0.1;
#endif
#ifdef IS_KIWI
        } else if(CHECK_MATERIAL(hit.material, M_KIWI)) {
            hit.albedo = kiwiTexture(newHit.uv).rgb + 0.1 * skybox(reflected).rgb + height * 0.1;
#endif
#ifdef IS_PEACH
        } else if(CHECK_MATERIAL(hit.material, M_PEACH)) {
            hit.albedo = peachTexture(newHit.uv).albedo + 0.1 * skybox(reflected).rgb + height * 0.1;
#endif
        }
    }

    Hit backside = march(hit.position + rayDirection * backsideRayDirectionStep, backsideSide * rayDirection, -backsideSide);
    float depth = length(backside.position - hit.position);

#ifdef IS_GRAPE
    if(CHECK_MATERIAL(hit.material, M_GRAPE)) {
        hit.albedo = mix(hit.albedo, grapeTexture(backside.uv).albedo, vec3(0.3));
    }
#endif

    vec3 color = vec3(0.);

    float shade = softshadow(hit.position, light2Direction, .1, 10., 32.);

    hit.emissive += pow(Z(1. - depth), 1.) * 0.25 * fakeSSSAmount;

    if(CHECK_MATERIAL(hit.material, M_BG)) {
        shade = 0.8 + shade * 0.2;
        hit.emissive = 0.;
    }

    color = .5 * fancyLighting(hit, light1Direction, normal, -rayDirection, 1.);
    color += .95 * fancyLighting(hit, light2Direction, normal, -rayDirection, shade);

    float ambient = .3;
    color = color * (1. - ambient) + ambient * hit.albedo;

    color = mix(bg, color, 1. - Z((hit.distance - 40.) / 10.));

    return color;

}


void main() {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 iResolution = vec2(resolutionX, resolutionY);
    vec2 uv = gl_FragCoord.xy/iResolution.xy;

    uv -= 0.5;


    vec3 pixel = vec3(1. / iResolution.xy / 4., 0.);

#ifdef AA
    vec3 color = image(uv + pixel.xz) + image(uv - pixel.xz) + image(uv + pixel.zy) + image(uv - pixel.zy);
    color *= 0.25;
#else
    vec3 color = image(uv);
#endif

    color *= smosh(378. - 200., frame, 200.);

    gl_FragColor = vec4(color, 1.);

}
