using RandomFields
using Documenter

DocMeta.setdocmeta!(RandomFields, :DocTestSetup, :(using RandomFields); recursive=true)

makedocs(;
    modules=[RandomFields],
    authors="Matt Graham <m.graham@ucl.ac.uk> and contributors",
    sitename="RandomFields.jl",
    format=Documenter.HTML(;
        canonical="https://aria-verify.github.io/RandomFields.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=Any[
        "Home" => "index.md",
        "API reference" => ["Public" => "lib/public.md", "Internals" => "lib/internals.md"],
    ],
)

deploydocs(; repo="github.com/aria-verify/RandomFields.jl", devbranch="main")
