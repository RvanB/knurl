"""Realistic host-language test with GLSL shader sources in Python strings."""

from OpenGL.GL import GL_FRAGMENT_SHADER, GL_VERTEX_SHADER


VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coordinates;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 fragment_texture_coordinates;

void main() {
    fragment_texture_coordinates = texture_coordinates;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""


FRAGMENT_SHADER = """#version 330 core
in vec2 fragment_texture_coordinates;

uniform sampler2D diffuse_texture;
uniform vec3 tint;
uniform float exposure;

out vec4 fragment_color;

vec3 tone_map(vec3 color) {
    return vec3(1.0) - exp(-color * exposure);
}

void main() {
    vec4 sampled_color = texture(diffuse_texture, fragment_texture_coordinates);
    vec3 mapped_color = tone_map(sampled_color.rgb * tint);
    fragment_color = vec4(mapped_color, sampled_color.a);
}
"""


def build_shader_program(compile_shader, link_program):
    """Compile the embedded GLSL and link it into a rendering program."""
    vertex = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    return link_program(vertex, fragment)
