//! Native entry point. Runs the same rendering core as the browser build.

fn main() {
    pollster::block_on(trd::run());
}
