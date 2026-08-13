---
layout: page
permalink: /repositories/
title: Repositories
description:
body_class: page-repositories
nav: true
nav_order: 4
---

<div class="repo-page">
  <section class="repo-section">
    <div class="repo-section-head">
      <div>
        <h2>GitHub Profile</h2>
        <p class="repo-section-note">Rendered as native cards instead of third-party image widgets.</p>
      </div>
    </div>

    {% if site.data.repositories.github_users %}
      <div class="repo-user-grid">
        {% for user in site.data.repositories.github_users %}
          <article class="github-user-card repo-skeleton" data-github-user="{{ user }}">
            <a class="repo-card-link" href="https://github.com/{{ user }}">
              <div class="repo-user-top">
                <div class="repo-user-avatar"></div>
                <div class="repo-user-meta">
                  <h3>{{ user }}</h3>
                  <p>@{{ user }}</p>
                </div>
              </div>
              <p class="repo-user-bio">Loading profile summary...</p>
              <div class="repo-stat-row">
                <span class="repo-stat-pill">Repos</span>
                <span class="repo-stat-pill">Followers</span>
                <span class="repo-stat-pill">Following</span>
              </div>
            </a>
          </article>
        {% endfor %}
      </div>
    {% endif %}

  </section>

  <section class="repo-section">
    <div class="repo-section-head">
      <div>
        <h2>Featured Repositories</h2>
        <p class="repo-section-note">Pinned as compact HTML cards with live stars, forks, language, and last update.</p>
      </div>
    </div>

    {% if site.data.repositories.github_repos %}
      <div class="repo-grid">
        {% for repo in site.data.repositories.github_repos %}
          {% assign repo_url = repo | split: '/' %}
          <article
            class="repo-card repo-skeleton"
            data-github-repo="{{ repo }}"
            data-github-owner="{{ repo_url[0] }}"
            data-github-name="{{ repo_url[1] }}"
          >
            <a class="repo-card-link" href="https://github.com/{{ repo }}">
              <div class="repo-card-head">
                <span class="repo-owner">{{ repo_url[0] }}</span>
                <span class="repo-chip">GitHub</span>
              </div>
              <h3>{{ repo_url[1] }}</h3>
              <p class="repo-description">Loading repository details...</p>
              <div class="repo-stat-row">
                <span class="repo-stat-pill">Stars</span>
                <span class="repo-stat-pill">Forks</span>
                <span class="repo-stat-pill">Updated</span>
              </div>
            </a>
          </article>
        {% endfor %}
      </div>
    {% endif %}

  </section>
</div>

<script defer src="{{ '/assets/js/repositories.js' | relative_url | bust_file_cache }}"></script>
