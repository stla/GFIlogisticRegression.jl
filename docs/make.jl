push!(LOAD_PATH, "../src/")

using Documenter
using GFIlogisticRegression

makedocs(
    sitename = "GFIlogisticRegression",
    authors = "Stéphane Laurent",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://stla.github.io/GFIlogisticRegression.jl",
        assets = String[],
    ),
    modules = [GFIlogisticRegression],
    pages = ["Documentation"  => "index.md"],
    repo = "https://github.com/stla/GFIlogisticRegression.jl/blob/{commit}{path}#{line}"
)

deploydocs(;
    branch = "gh-pages",
    devbranch = "main",
    repo = "github.com/stla/GFIlogisticRegression.jl",
)