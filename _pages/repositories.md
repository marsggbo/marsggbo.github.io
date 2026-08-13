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
  <div class="repo-showcase">
    <section class="repo-hero">
      <div class="repo-hero-copy">
        <span class="repo-kicker">Open Source</span>
        <h2>{{ site.data.repositories.profile.name }} on GitHub</h2>
        <p class="repo-hero-text">{{ site.data.repositories.profile.bio }}</p>
        <div class="repo-hero-meta">
          <span>{{ site.data.repositories.profile.location }}</span>
          <span>{{ site.data.repositories.profile.public_repos }} public repos</span>
          <span>Research code + tools + notes</span>
        </div>
        <div class="repo-hero-actions">
          <a class="repo-cta" href="{{ site.data.repositories.profile.url }}" target="_blank" rel="noopener noreferrer">
            <i class="fa-brands fa-github"></i>
            <span>View Profile</span>
          </a>
          <a class="repo-ghost" href="{{ site.data.repositories.profile.website }}" target="_blank" rel="noopener noreferrer">
            <span>Website</span>
          </a>
        </div>
      </div>

      <div class="repo-hero-card">
        <img class="repo-hero-avatar" src="{{ site.data.repositories.profile.avatar }}" alt="{{ site.data.repositories.profile.username }}">
        <div class="repo-hero-stats">
          <div class="repo-hero-stat">
            <strong>{{ site.data.repositories.profile.public_repos }}</strong>
            <span>Repos</span>
          </div>
          <div class="repo-hero-stat">
            <strong>{{ site.data.repositories.profile.followers }}</strong>
            <span>Followers</span>
          </div>
          <div class="repo-hero-stat">
            <strong>{{ site.data.repositories.profile.following }}</strong>
            <span>Following</span>
          </div>
        </div>
      </div>
    </section>

    <section class="repo-shelf">
      <div class="repo-shelf-head">
        <div>
          <span class="repo-kicker">Flagship</span>
          <h3>Flagship Repos</h3>
        </div>
        <p>Higher-signal work: frameworks, paper codebases, and projects that represent the strongest technical taste.</p>
      </div>

      <div class="repo-curated-grid repo-curated-grid-featured">
        {% for repo in site.data.repositories.featured_repos %}
          <article class="repo-curated-card" data-language="{{ repo.language | slugify }}">
            <div class="repo-curated-top">
              <span class="repo-kind">{{ repo.kind }}</span>
              <span class="repo-language">{{ repo.language }}</span>
            </div>
            <h4>{{ repo.name }}</h4>
            <p class="repo-curated-name">{{ repo.full_name }}</p>
            <p class="repo-curated-summary">{{ repo.summary }}</p>
            {% if repo.topics %}
              <div class="repo-topic-row">
                {% for topic in repo.topics limit: 3 %}
                  <span class="repo-topic">{{ topic }}</span>
                {% endfor %}
              </div>
            {% endif %}
            <div class="repo-curated-meta">
              <span><i class="fa-solid fa-star fa-sm"></i> {{ repo.stars }}</span>
              <span><i class="fa-solid fa-code-fork fa-sm"></i> {{ repo.forks }}</span>
              <span>{{ repo.updated_at | date: "%b %Y" }}</span>
            </div>
            <div class="repo-curated-actions">
              <a class="repo-cta repo-cta-small" href="{{ repo.url }}" target="_blank" rel="noopener noreferrer">Repository</a>
              {% if repo.external_url %}
                <a class="repo-ghost repo-ghost-small" href="{{ repo.external_url }}" target="_blank" rel="noopener noreferrer">{{ repo.external_label }}</a>
              {% endif %}
            </div>
          </article>
        {% endfor %}
      </div>
    </section>

    <section class="repo-shelf">
      <div class="repo-shelf-head">
        <div>
          <span class="repo-kicker">Lab Shelf</span>
          <h3>Research And Experiments</h3>
        </div>
        <p>Smaller utilities, earlier experiments, and paper-adjacent repos that still show the evolution of the work.</p>
      </div>

      <div class="repo-curated-grid repo-curated-grid-compact">
        {% for repo in site.data.repositories.lab_repos %}
          <article class="repo-curated-card repo-curated-card-compact" data-language="{{ repo.language | slugify }}">
            <div class="repo-curated-top">
              <span class="repo-kind">{{ repo.kind }}</span>
              <span class="repo-language">{{ repo.language }}</span>
            </div>
            <h4>{{ repo.name }}</h4>
            <p class="repo-curated-summary">{{ repo.summary }}</p>
            {% if repo.topics %}
              <div class="repo-topic-row">
                {% for topic in repo.topics limit: 3 %}
                  <span class="repo-topic">{{ topic }}</span>
                {% endfor %}
              </div>
            {% endif %}
            <div class="repo-curated-meta">
              <span><i class="fa-solid fa-star fa-sm"></i> {{ repo.stars }}</span>
              <span><i class="fa-solid fa-code-fork fa-sm"></i> {{ repo.forks }}</span>
              <span>{{ repo.updated_at | date: "%Y" }}</span>
            </div>
            <div class="repo-curated-actions">
              <a class="repo-cta repo-cta-small" href="{{ repo.url }}" target="_blank" rel="noopener noreferrer">Open</a>
              {% if repo.external_url %}
                <a class="repo-ghost repo-ghost-small" href="{{ repo.external_url }}" target="_blank" rel="noopener noreferrer">{{ repo.external_label }}</a>
              {% endif %}
            </div>
          </article>
        {% endfor %}
      </div>
    </section>

  </div>
</div>
