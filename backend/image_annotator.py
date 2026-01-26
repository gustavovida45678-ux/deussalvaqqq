"""
Image Annotator for Trading Chart Analysis
Adds visual annotations to trading charts based on AI analysis
"""
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Optional
import re


class ChartAnnotator:
    """Annotates trading charts with entry/exit points and analysis"""
    
    def __init__(self):
        self.colors = {
            'call': '#00ff00',  # Green for CALL/BUY
            'put': '#ff0000',   # Red for PUT/SELL
            'support': '#0088ff',  # Blue for support
            'resistance': '#ff8800',  # Orange for resistance
            'text_bg': '#000000cc',  # Semi-transparent black
            'text_fg': '#ffffff',  # White text
            'arrow': '#ffff00',  # Yellow arrows
        }
    
    def extract_trading_signals(self, analysis_text: str) -> Dict:
        """
        Extract trading signals from AI analysis text
        Returns dict with entry points, exit points, strategy, etc.
        """
        signals = {
            'action': None,  # 'CALL', 'PUT', or 'WAIT'
            'entry_price': None,
            'exit_price': None,
            'stop_loss': None,
            'take_profit': None,
            'strategy': None,
            'confidence': None,
            'key_levels': [],
        }
        
        text_upper = analysis_text.upper()
        
        # Detect action (CALL/PUT/BUY/SELL)
        if any(word in text_upper for word in ['COMPRA', 'CALL', 'BUY', 'ALTA']):
            signals['action'] = 'CALL'
        elif any(word in text_upper for word in ['VENDA', 'PUT', 'SELL', 'BAIXA']):
            signals['action'] = 'PUT'
        elif any(word in text_upper for word in ['AGUARDAR', 'WAIT', 'NEUTRO']):
            signals['action'] = 'WAIT'
        
        # Extract confidence level
        confidence_match = re.search(r'(\d+)%.*confiança', text_upper)
        if confidence_match:
            signals['confidence'] = int(confidence_match.group(1))
        
        # Extract strategy type
        strategy_patterns = [
            'COUNTER-TREND', 'TREND-FOLLOWING', 'BREAKOUT',
            'REVERSAL', 'PULLBACK', 'CONTINUAÇÃO', 'REVERSÃO'
        ]
        for pattern in strategy_patterns:
            if pattern in text_upper:
                signals['strategy'] = pattern
                break
        
        # Extract price levels (simplified - could be improved with better parsing)
        price_matches = re.findall(r'(\d+[.,]\d+)', analysis_text)
        if price_matches:
            signals['key_levels'] = [float(p.replace(',', '.')) for p in price_matches[:5]]
        
        return signals
    
    def annotate_chart(self, 
                       image_bytes: bytes, 
                       analysis_text: str,
                       signals: Optional[Dict] = None) -> bytes:
        """
        Add annotations to trading chart based on analysis
        Returns annotated image as bytes
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image, 'RGBA')
        width, height = image.size
        
        # Extract signals if not provided
        if signals is None:
            signals = self.extract_trading_signals(analysis_text)
        
        # Try to load a font, fallback to default if not available
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Add main recommendation banner
        if signals['action'] in ['CALL', 'PUT']:
            self._draw_recommendation_banner(
                draw, width, height, 
                signals['action'], 
                signals.get('confidence'),
                font_large
            )
        
        # Add entry point annotation
        if signals['action'] == 'CALL':
            self._draw_entry_annotation(
                draw, width, height,
                'CALL ENTRY',
                self.colors['call'],
                position='bottom',
                font=font_medium
            )
        elif signals['action'] == 'PUT':
            self._draw_entry_annotation(
                draw, width, height,
                'PUT ENTRY',
                self.colors['put'],
                position='top',
                font=font_medium
            )
        
        # Add strategy label if available
        if signals['strategy']:
            self._draw_strategy_label(
                draw, width, height,
                signals['strategy'],
                font_small
            )
        
        # Add key levels if available
        if signals['key_levels'] and len(signals['key_levels']) > 0:
            self._draw_key_levels(
                draw, width, height,
                signals['key_levels'],
                font_small
            )
        
        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()
    
    def _draw_recommendation_banner(self, draw, width, height, action, confidence, font):
        """Draw main recommendation banner at top"""
        text = f"{action} {'📈' if action == 'CALL' else '📉'}"
        if confidence:
            text += f" ({confidence}% confiança)"
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at top center
        x = (width - text_width) // 2
        y = 20
        
        # Draw background
        color = self.colors['call'] if action == 'CALL' else self.colors['put']
        padding = 15
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 200)
        )
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            outline=color,
            width=3
        )
        
        # Draw text
        draw.text((x, y), text, fill=color, font=font)
    
    def _draw_entry_annotation(self, draw, width, height, label, color, position, font):
        """Draw entry point annotation with arrow"""
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position based on entry type
        if position == 'bottom':
            x = width // 3
            y = height - 100
            arrow_y = y - 30
        else:  # top
            x = width // 3
            y = 100
            arrow_y = y + text_height + 30
        
        # Draw background box
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((x, y), label, fill=color, font=font)
        
        # Draw arrow
        arrow_points = [
            (x + text_width // 2 - 10, arrow_y),
            (x + text_width // 2 + 10, arrow_y),
            (x + text_width // 2, arrow_y + (20 if position == 'bottom' else -20))
        ]
        draw.polygon(arrow_points, fill=color)
    
    def _draw_strategy_label(self, draw, width, height, strategy, font):
        """Draw strategy type label"""
        label = f"Estratégia: {strategy}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = 20
        y = height - 50
        
        # Draw background
        padding = 8
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 150)
        )
        
        # Draw text
        draw.text((x, y), label, fill='#ffffff', font=font)
    
    def _draw_key_levels(self, draw, width, height, levels, font):
        """Draw key support/resistance levels"""
        # Draw first 3 levels as horizontal lines with labels
        for i, level in enumerate(levels[:3]):
            y = height // 4 + (i * height // 6)
            
            # Draw line
            color = self.colors['resistance'] if i % 2 == 0 else self.colors['support']
            draw.line([(50, y), (width - 50, y)], fill=color, width=2)
            
            # Draw label
            label = f"{level:.4f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            
            padding = 5
            draw.rectangle(
                [width - text_width - 60, y - 12, width - 55, y + 12],
                fill=(0, 0, 0, 180)
            )
            draw.text((width - text_width - 58, y - 10), label, fill=color, font=font)


def create_annotated_chart(image_bytes: bytes, analysis_text: str) -> bytes:
    """
    Convenience function to create annotated chart
    """
    annotator = ChartAnnotator()
    return annotator.annotate_chart(image_bytes, analysis_text)
