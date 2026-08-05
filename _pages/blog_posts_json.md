---
layout: none
permalink: /blog/posts.json
---
[
  {%- assign posts = site.posts | where_exp: "p", "p.redirect == nil" -%}
  {%- for post in posts -%}
    {%- assign primary = post.tags | first | default: "未分类" -%}
    {
      "title": {{ post.title | jsonify }},
      "url": {{ post.url | relative_url | jsonify }},
      "date": {{ post.date | date: "%Y-%m-%d" | jsonify }},
      "year": {{ post.date | date: "%Y" | jsonify }},
      "desc": {{ post.description | default: "" | strip_html | truncate: 120 | jsonify }},
      "tags": {{ post.tags | jsonify }},
      "topic": {{ primary | jsonify }}
    }{%- unless forloop.last -%},{%- endunless -%}
  {%- endfor -%}
]
