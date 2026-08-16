require 'fileutils'
require 'set'
require 'shellwords'

module SelectiveWebp
  class GeneratedImageFile < Jekyll::StaticFile
    def write(_dest)
      true
    end
  end

  class Generator < Jekyll::Generator
    DEFAULTS = {
      'enabled' => true,
      'widths' => [480, 800, 1400],
      'input_directories' => ['assets/img/'],
      'input_formats' => ['.jpg', '.jpeg', '.png', '.tiff', '.gif'],
      'auto_formats' => ['.png', '.tiff', '.gif'],
      'output_formats' => { 'webp' => '-quality 85' },
      'exclude' => [],
      'include' => [],
      'min_source_bytes' => 1_048_576
    }.freeze

    safe true
    priority :lowest

    def generate(site)
      @site = site
      @config = DEFAULTS.merge(site.config['selective_webp'] || {})
      site.data['generated_webp'] = {}

      return unless @config['enabled']

      ensure_dest_exists
      generate_webp_files
    end

    private

    def ensure_dest_exists
      FileUtils.mkdir_p(@site.dest) unless File.directory?(@site.dest)
    end

    def generate_webp_files
      generated = 0

      candidate_files.each do |input_file|
        next unless selected_for_webp?(input_file)

        logical_path = normalize_path(input_file.sub(@site.source, ''))
        @site.data['generated_webp'][logical_path] = true

        @config['output_formats'].each do |format_extension, flags|
          @config['widths'].each do |edge|
            output_file = output_path_for(input_file, format_extension, edge)
            FileUtils.mkdir_p(File.dirname(output_file))

            if stale_output?(input_file, output_file)
              convert_image(input_file, output_file, edge, flags)
              generated += 1
            end

            next unless File.file?(output_file)

            @site.static_files << GeneratedImageFile.new(
              @site,
              @site.dest,
              File.dirname(logical_path),
              File.basename(output_file)
            )
          end
        end
      end

      Jekyll.logger.info('SelectiveWebp:', "Generated #{generated} file(s)")
    end

    def candidate_files
      files = []
      formats = @config['input_formats'].map(&:downcase).to_set

      @config['input_directories'].each do |directory|
        Dir[File.join(@site.source, directory, '**', '*.*')].each do |full_path|
          next unless formats.include?(File.extname(full_path).downcase)

          files << full_path
        end
      end

      files
    end

    def selected_for_webp?(input_file)
      logical_path = normalize_path(input_file.sub(@site.source, ''))
      return false if matches_any_pattern?(logical_path, @config['exclude'])
      return true if matches_any_pattern?(logical_path, @config['include'])

      extension = File.extname(input_file).downcase
      return false unless Array(@config['auto_formats']).map(&:downcase).include?(extension)

      File.size(input_file) >= @config['min_source_bytes'].to_i
    end

    def output_path_for(input_file, format_extension, edge)
      input_ext = File.extname(input_file)
      prefix = File.dirname(input_file.sub(@site.source, ''))
      suffix = edge.to_i.zero? ? '' : "-#{edge}"
      filename = "#{File.basename(input_file, input_ext)}#{suffix}.#{format_extension}"
      File.join(@site.dest, prefix, filename)
    end

    def stale_output?(input_file, output_file)
      !File.file?(output_file) || File.mtime(output_file) <= File.mtime(input_file)
    end

    def convert_image(input_file, output_file, edge, flags)
      unless input_file.start_with?(@site.source) && output_file.start_with?(@site.dest)
        raise "refusing to convert paths outside site source/dest: #{input_file} -> #{output_file}"
      end

      cmd = converter_binary
      resize = edge.to_i.zero? ? [] : ['-resize', "#{edge}>"]
      extra_flags = Shellwords.split(flags.to_s)
      args = [cmd, input_file, *resize, *extra_flags, output_file]

      success = system(*args)
      return if success

      raise "Failed to generate #{output_file} from #{input_file}"
    end

    def converter_binary
      return 'magick' if system('which', 'magick', out: File::NULL, err: File::NULL)
      return 'convert' if system('which', 'convert', out: File::NULL, err: File::NULL)

      raise 'ImageMagick binary not found. Expected `magick` or `convert` in PATH.'
    end

    def matches_any_pattern?(logical_path, patterns)
      Array(patterns).any? do |pattern|
        File.fnmatch?(pattern, logical_path, File::FNM_PATHNAME | File::FNM_EXTGLOB)
      end
    end

    def normalize_path(path)
      path.sub(%r{\A/+}, '')
    end
  end
end
