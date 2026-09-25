import AppKit
import Foundation
import PDFKit

let arguments = CommandLine.arguments
guard arguments.count >= 4,
      let pageNumber = Int(arguments[2]),
      let document = PDFDocument(url: URL(fileURLWithPath: arguments[1])),
      let page = document.page(at: pageNumber - 1) else {
    fputs("Usage: render_pages <input.pdf> <page-number> <output.png>\n", stderr)
    exit(1)
}

let bounds = page.bounds(for: .mediaBox)
let scale: CGFloat = 2
let width = Int(bounds.width * scale)
let height = Int(bounds.height * scale)
guard let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
) else { exit(1) }

guard let context = NSGraphicsContext(bitmapImageRep: bitmap) else { exit(1) }
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.cgContext.setFillColor(NSColor.white.cgColor)
context.cgContext.fill(CGRect(x: 0, y: 0, width: width, height: height))
context.cgContext.scaleBy(x: scale, y: scale)
page.draw(with: .mediaBox, to: context.cgContext)
NSGraphicsContext.restoreGraphicsState()

let output = URL(fileURLWithPath: arguments[3])
try bitmap.representation(using: .png, properties: [:])?.write(to: output)
