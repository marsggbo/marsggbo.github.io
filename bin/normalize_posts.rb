#!/usr/bin/env ruby

require "cgi"
require "date"
require "uri"
require "yaml"

POST_GLOB = "_posts/*.md"
FORMULA_IMAGE_RE = /!\[(.*?)\]\((https:\/\/www\.zhihu\.com\/equation\?tex=[^)]+|\/assets\/img\/marsggbo\/[^)]+)\)/
INLINE_MATH_RE = /(?<!\\)(\$\$?)(.+?)(?<!\\)\1/m

def read_post(path)
  content = File.read(path, encoding: "UTF-8")
  lines = content.lines
  return nil unless lines.first&.strip == "---"

  boundary = lines[1..]&.index { |line| line.strip == "---" }
  return nil unless boundary

  front_matter = lines[1..boundary].join
  body = lines[(boundary + 2)..]&.join.to_s
  [front_matter, body]
end

def dump_scalar(value)
  YAML.dump(value)
    .sub(/\A---(?:\s*\n|\s+)/, "")
    .sub(/\n\.\.\.\s*\z/, "")
    .strip
end

def inline_array?(value)
  value.all? { |item| item.is_a?(String) || item.is_a?(Numeric) || item == true || item == false }
end

def dump_key_value(key, value, indent = 0)
  prefix = " " * indent

  case value
  when Hash
    lines = ["#{prefix}#{key}:"]
    value.each do |sub_key, sub_value|
      lines << dump_key_value(sub_key.to_s, sub_value, indent + 2)
    end
    lines.join("\n")
  when Array
    if value.empty?
      "#{prefix}#{key}: []"
    elsif inline_array?(value)
      "#{prefix}#{key}: [#{value.map { |item| dump_scalar(item) }.join(', ')}]"
    else
      lines = ["#{prefix}#{key}:"]
      value.each do |item|
        if item.is_a?(Hash)
          lines << "#{prefix}-"
          item.each do |sub_key, sub_value|
            lines << dump_key_value(sub_key.to_s, sub_value, indent + 2)
          end
        else
          lines << "#{prefix}- #{dump_scalar(item)}"
        end
      end
      lines.join("\n")
    end
  else
    "#{prefix}#{key}: #{dump_scalar(value)}"
  end
end

def normalize_tags(value)
  case value
  when nil
    []
  when Array
    value.map { |item| sanitize_value(item) }
  when String
    stripped = value.strip.sub(/\A---\s+/, "")
    if stripped.start_with?("[") && stripped.end_with?("]")
      parsed = YAML.safe_load(stripped, aliases: true)
      parsed.is_a?(Array) ? parsed.map { |item| sanitize_value(item) } : [stripped]
    elsif stripped.include?(",")
      stripped.split(/\s*,\s*/)
    else
      stripped.split(/\s+/)
    end
  else
    [value.to_s]
  end.map(&:to_s).map(&:strip).map { |item| item.sub(/\A\/+/, "") }.reject(&:empty?)
end

def sanitize_value(value)
  case value
  when String
    value.sub(/\A---\s+/, "")
  when Array
    value.map { |item| sanitize_value(item) }
  when Hash
    value.transform_values { |item| sanitize_value(item) }
  else
    value
  end
end

def strip_wrapping_quotes(value)
  return value unless value.is_a?(String)
  return value unless value.length >= 2

  if (value.start_with?('"') && value.end_with?('"')) || (value.start_with?("'") && value.end_with?("'"))
    value[1..-2]
  else
    value
  end
end

def filename_date(path)
  File.basename(path)[/\A\d{4}-\d{2}-\d{2}/]
end

def normalize_front_matter(front_matter, path)
  data = YAML.safe_load(front_matter, permitted_classes: [Date, Time], aliases: true) || {}
  return front_matter unless data.is_a?(Hash)

  normalized = {}
  normalized["layout"] = sanitize_value(data["layout"]).to_s
  normalized["layout"] = "post" if normalized["layout"].empty?
  normalized["title"] = strip_wrapping_quotes(sanitize_value(data["title"]).to_s)

  date_value = data["date"]
  normalized["date"] = filename_date(path) ||
    case date_value
    when Date, Time, DateTime
      date_value.strftime("%Y-%m-%d")
    when String
      sanitize_value(date_value).to_s.strip[0, 10]
    else
      data["date"].to_s
    end

  tags = normalize_tags(data["tags"])
  tags = normalize_tags(data["categories"]) if tags.empty? && data.key?("categories")
  tags = normalize_tags(data["category"]) if tags.empty? && data.key?("category")
  normalized["tags"] = tags

  data.each do |key, value|
    next if %w[layout title date tags].include?(key.to_s)
    normalized[key.to_s] = sanitize_value(value)
    normalized[key.to_s] = normalized[key.to_s].sub(/\A\/+/, "") if %w[category categories].include?(key.to_s) && normalized[key.to_s].is_a?(String)
    if %w[grammar_cjkRuby related_posts published].include?(key.to_s) && normalized[key.to_s].is_a?(String)
      if normalized[key.to_s].casecmp("true").zero?
        normalized[key.to_s] = true
      elsif normalized[key.to_s].casecmp("false").zero?
        normalized[key.to_s] = false
      end
    end
  end

  lines = ["---"]
  normalized.each do |key, value|
    lines << dump_key_value(key, value)
  end
  lines << "---"
  lines.join("\n") + "\n"
end

def standalone_formula?(content, match)
  line_start = content.rindex("\n", match.begin(0) - 1)
  line_start = line_start ? line_start + 1 : 0
  line_end = content.index("\n", match.end(0)) || content.length
  line = content[line_start...line_end]
  line.strip == match[0].strip
end

def decode_zhihu_formula(url)
  URI.parse(url).query.to_s.split("&").each do |pair|
    key, value = pair.split("=", 2)
    return CGI.unescape(value.to_s) if key == "tex"
  end
  nil
rescue URI::InvalidURIError
  nil
end

def looks_like_formula?(text)
  stripped = text.to_s.strip
  return false if stripped.empty?

  stripped.match?(/[\\_^={}]/) ||
    stripped.match?(/\b(?:sum|frac|sqrt|theta|alpha|beta|gamma|mathbb|mathcal|begin|end|left|right|cdot|times)\b/i) ||
    stripped.include?("=")
end

def cleanup_formula(formula)
  cleaned = CGI.unescapeHTML(formula.to_s)
  cleaned = cleaned.gsub(/\r\n?/, "\n").strip
  cleaned = cleaned.gsub(/\\{2,}(?=[A-Za-z])/, "\\")
  cleaned = cleaned.sub(/\A\\\[/, "").sub(/\\\]\z/, "")
  cleaned = cleaned.sub(/\A\\\(/, "").sub(/\\\)\z/, "")
  cleaned = cleaned.sub(/\A\[\s*(.+?)\s*\]\z/m, "\\1") if cleaned.start_with?("[[") || cleaned.end_with?("]]")
  cleaned = cleaned.sub(/(?:\\\\)+\z/, "").strip

  loop do
    collapsed = cleaned.gsub("{{", "{").gsub("}}", "}")
    break if collapsed == cleaned
    cleaned = collapsed
  end

  cleaned
end

def replace_formula_images(body)
  body.gsub(FORMULA_IMAGE_RE) do
    match = Regexp.last_match
    alt = match[1]
    url = match[2]

    formula =
      if url.start_with?("https://www.zhihu.com/equation?tex=")
        decode_zhihu_formula(url) || alt
      elsif looks_like_formula?(alt)
        alt
      end

    next match[0] if formula.nil? || formula.strip.empty?

    cleaned = cleanup_formula(formula)
    next match[0] if cleaned.empty?

    delimiter = standalone_formula?(body, match) ? "$$" : "$"
    "#{delimiter}#{cleaned}#{delimiter}"
  end
end

def cleanup_existing_math(body)
  body.gsub(INLINE_MATH_RE) do
    delimiter = Regexp.last_match(1)
    formula = Regexp.last_match(2)
    "#{delimiter}#{cleanup_formula(formula)}#{delimiter}"
  end
end

changed = 0

Dir.glob(POST_GLOB).sort.each do |path|
  parsed = read_post(path)
  next unless parsed

  front_matter, body = parsed
  new_front_matter = normalize_front_matter(front_matter, path)
  new_body = cleanup_existing_math(replace_formula_images(body))
  new_content = new_front_matter + new_body
  next if new_content == File.read(path, encoding: "UTF-8")

  File.write(path, new_content)
  changed += 1
end

puts "updated=#{changed}"
