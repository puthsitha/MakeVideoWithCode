import AppKit
import Foundation

// Usage: khmer_text_render <text> <fontSize> <maxWidth> <hexColor> <outPath>
guard CommandLine.arguments.count >= 6 else {
    fputs("Usage: khmer_text_render <text> <fontSize> <maxWidth> <hexColor> <outPath>\n", stderr)
    exit(1)
}

let text = CommandLine.arguments[1].replacingOccurrences(of: "\\n", with: "\n")
let fontSize = CGFloat(Double(CommandLine.arguments[2]) ?? 54.0)
let maxWidth = CGFloat(Double(CommandLine.arguments[3]) ?? 860.0)
let hexColor = CommandLine.arguments[4].trimmingCharacters(in: CharacterSet(charactersIn: "#"))
let outPath = CommandLine.arguments[5]

func colorFromHex(_ hex: String) -> NSColor {
    var rgb: UInt64 = 0
    Scanner(string: hex).scanHexInt64(&rgb)
    let r = CGFloat((rgb >> 16) & 0xFF) / 255.0
    let g = CGFloat((rgb >> 8) & 0xFF) / 255.0
    let b = CGFloat(rgb & 0xFF) / 255.0
    return NSColor(calibratedRed: r, green: g, blue: b, alpha: 1.0)
}

let font = NSFont(name: "KhmerMN-Bold", size: fontSize)
    ?? NSFont(name: "KhmerSangamMN", size: fontSize)
    ?? NSFont.boldSystemFont(ofSize: fontSize)

let para = NSMutableParagraphStyle()
para.alignment = .center
para.lineSpacing = fontSize * 0.22

let shadow = NSShadow()
shadow.shadowColor = NSColor(calibratedWhite: 0.0, alpha: 0.92)
shadow.shadowOffset = NSSize(width: 0, height: -3)
shadow.shadowBlurRadius = 10.0

let attrs: [NSAttributedString.Key: Any] = [
    .font: font,
    .foregroundColor: colorFromHex(hexColor),
    .paragraphStyle: para,
    .shadow: shadow,
]

let attrStr = NSAttributedString(string: text, attributes: attrs)
let padX: CGFloat = 44.0
let padY: CGFloat = 28.0
let measureRect = attrStr.boundingRect(
    with: NSSize(width: maxWidth - padX * 2, height: 2000),
    options: [.usesLineFragmentOrigin, .usesFontLeading]
)

let imgW = max(64, Int(ceil(measureRect.width + padX * 2)))
let imgH = max(64, Int(ceil(measureRect.height + padY * 2)))

guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: imgW,
    pixelsHigh: imgH,
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
) else {
    fputs("Failed to create bitmap\n", stderr)
    exit(2)
}

NSGraphicsContext.saveGraphicsState()
if let ctx = NSGraphicsContext(bitmapImageRep: rep) {
    NSGraphicsContext.current = ctx
    NSColor.clear.setFill()
    NSRect(x: 0, y: 0, width: imgW, height: imgH).fill()

    let drawRect = NSRect(
        x: padX,
        y: padY * 0.85,
        width: CGFloat(imgW) - padX * 2,
        height: CGFloat(imgH) - padY * 1.4
    )
    attrStr.draw(with: drawRect, options: [.usesLineFragmentOrigin, .usesFontLeading])
}
NSGraphicsContext.restoreGraphicsState()

guard let pngData = rep.representation(using: .png, properties: [:]) else {
    fputs("Failed to encode PNG\n", stderr)
    exit(3)
}

try pngData.write(to: URL(fileURLWithPath: outPath))
print("OK \(imgW)x\(imgH)")
