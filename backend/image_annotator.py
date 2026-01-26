"""
Image Annotator for Trading Chart Analysis
Adds visual annotations to trading charts based on AI analysis
Enhanced with OCR and advanced pattern detection
"""
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


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
            'trend_line': '#00ffff',  # Cyan for trend lines
            'zone': '#ff00ff44',  # Semi-transparent magenta for zones
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
            'support_levels': [],
            'resistance_levels': [],
            'trend': None,  # 'ALTA', 'BAIXA', 'LATERAL'
        }
        
        text_upper = analysis_text.upper()
        
        # Detect action (CALL/PUT/BUY/SELL)
        if any(word in text_upper for word in ['COMPRA', 'CALL', 'BUY', 'ALTA', 'BULLISH']):
            signals['action'] = 'CALL'
        elif any(word in text_upper for word in ['VENDA', 'PUT', 'SELL', 'BAIXA', 'BEARISH']):
            signals['action'] = 'PUT'
        elif any(word in text_upper for word in ['AGUARDAR', 'WAIT', 'NEUTRO', 'LATERAL']):
            signals['action'] = 'WAIT'
        
        # Detect trend
        if any(word in text_upper for word in ['TENDÊNCIA DE ALTA', 'UPTREND', 'ALTA']):
            signals['trend'] = 'ALTA'
        elif any(word in text_upper for word in ['TENDÊNCIA DE BAIXA', 'DOWNTREND', 'BAIXA']):
            signals['trend'] = 'BAIXA'
        elif any(word in text_upper for word in ['LATERAL', 'SIDEWAYS', 'CONSOLIDAÇÃO']):
            signals['trend'] = 'LATERAL'
        
        # Extract confidence level
        confidence_patterns = [
            r'(\d+)%.*(?:CONFIANÇA|CONFIDENCE)',
            r'(?:CONFIANÇA|CONFIDENCE).*?(\d+)%',
            r'NÍVEL DE CONFIANÇA.*?(\d+)%',
        ]
        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, text_upper)
            if confidence_match:
                signals['confidence'] = int(confidence_match.group(1))
                break
        
        # Extract strategy type
        strategy_patterns = [
            'COUNTER-TREND', 'CONTRA-TENDÊNCIA', 'TREND-FOLLOWING', 'SEGUIR TENDÊNCIA',
            'BREAKOUT', 'ROMPIMENTO', 'REVERSAL', 'REVERSÃO', 'PULLBACK', 'CONTINUAÇÃO'
        ]
        for pattern in strategy_patterns:
            if pattern in text_upper:
                signals['strategy'] = pattern
                break
        
        # Extract price levels (improved regex)
        price_matches = re.findall(r'(\d+[.,]\d+)', analysis_text)
        if price_matches:
            prices = [float(p.replace(',', '.')) for p in price_matches[:10]]
            signals['key_levels'] = sorted(set(prices))[:5]  # Remove duplicates and limit to 5
        
        # Extract support levels
        support_patterns = [
            r'SUPORTE.*?(\d+[.,]\d+)',
            r'SUPPORT.*?(\d+[.,]\d+)',
            r'REGIÃO.*?(\d+[.,]\d+)',
            r'ABAIXO.*?DE.*?(\d+[.,]\d+)',
        ]
        for pattern in support_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                signals['support_levels'].extend([float(m.replace(',', '.')) for m in matches])
        
        # Extract resistance levels
        resistance_patterns = [
            r'RESISTÊNCIA.*?(\d+[.,]\d+)',
            r'RESISTANCE.*?(\d+[.,]\d+)',
            r'ROMPER.*?(\d+[.,]\d+)',
            r'ACIMA.*?DE.*?(\d+[.,]\d+)',
        ]
        for pattern in resistance_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                signals['resistance_levels'].extend([float(m.replace(',', '.')) for m in matches])
        
        # Extract stop loss - improved patterns
        sl_patterns = [
            r'STOP.*?LOSS.*?(\d+[.,]\d+)',
            r'STOP.*?ABAIXO.*?(\d+[.,]\d+)',
            r'STOP.*?ACIMA.*?(\d+[.,]\d+)',
            r'SL.*?(\d+[.,]\d+)',
        ]
        for pattern in sl_patterns:
            sl_match = re.search(pattern, text_upper)
            if sl_match:
                signals['stop_loss'] = float(sl_match.group(1).replace(',', '.'))
                break
        
        # Extract take profit - improved patterns  
        tp_patterns = [
            r'TAKE.*?PROFIT.*?(\d+[.,]\d+)',
            r'ALVO[S]?.*?(\d+[.,]\d+)',
            r'TARGET.*?(\d+[.,]\d+)',
            r'TP.*?(\d+[.,]\d+)',
        ]
        for pattern in tp_patterns:
            tp_match = re.search(pattern, text_upper)
            if tp_match:
                signals['take_profit'] = float(tp_match.group(1).replace(',', '.'))
                break
        
        return signals
    
    def detect_chart_regions(self, image_bytes: bytes) -> Dict:
        """
        Detect chart regions using image processing
        Returns coordinates for different chart areas
        """
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        
        # Simple heuristic-based region detection
        regions = {
            'chart_area': {
                'x': int(width * 0.1),
                'y': int(height * 0.1),
                'width': int(width * 0.8),
                'height': int(height * 0.7),
            },
            'top_area': {
                'y_start': 0,
                'y_end': int(height * 0.15),
            },
            'bottom_area': {
                'y_start': int(height * 0.85),
                'y_end': height,
            },
            'left_area': {
                'x_start': 0,
                'x_end': int(width * 0.15),
            },
            'right_area': {
                'x_start': int(width * 0.85),
                'x_end': width,
            },
        }
        
        return regions
    
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
        
        # Convert to RGBA for transparency support
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create overlay for transparent elements
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Create main draw object
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Extract signals if not provided
        if signals is None:
            signals = self.extract_trading_signals(analysis_text)
        
        # Detect chart regions
        try:
            regions = self.detect_chart_regions(image_bytes)
        except Exception as e:
            logger.warning(f"Could not detect chart regions: {e}")
            regions = None
        
        # Try to load fonts, fallback to default if not available
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw zones first (background elements)
        if signals['trend'] and regions:
            self._draw_trend_zones(draw_overlay, width, height, signals['trend'], regions)
        
        # Draw support and resistance lines
        if signals['support_levels'] or signals['resistance_levels']:
            self._draw_support_resistance_lines(
                draw, width, height, 
                signals['support_levels'], 
                signals['resistance_levels'],
                font_small
            )
        
        # Draw trend lines
        if signals['trend']:
            self._draw_trend_lines(draw, width, height, signals['trend'])
        
        # Add main recommendation banner
        if signals['action'] in ['CALL', 'PUT']:
            self._draw_recommendation_banner(
                draw, width, height, 
                signals['action'], 
                signals.get('confidence'),
                font_large
            )
        
        # Add entry point annotation with improved positioning
        if signals['action'] == 'CALL':
            self._draw_entry_annotation_v2(
                draw, width, height,
                'CALL ENTRY',
                self.colors['call'],
                position='bottom-left',
                font=font_medium
            )
        elif signals['action'] == 'PUT':
            self._draw_entry_annotation_v2(
                draw, width, height,
                'PUT ENTRY',
                self.colors['put'],
                position='top-left',
                font=font_medium
            )
        
        # Add exit points if available
        if signals.get('take_profit'):
            self._draw_exit_annotation(
                draw, width, height,
                f"TP: {signals['take_profit']:.4f}",
                self.colors['call'] if signals['action'] == 'CALL' else self.colors['put'],
                font=font_small
            )
        
        if signals.get('stop_loss'):
            self._draw_stop_loss_annotation(
                draw, width, height,
                f"SL: {signals['stop_loss']:.4f}",
                '#ff0000',
                font=font_small
            )
        
        # Add strategy label if available
        if signals['strategy']:
            self._draw_strategy_label(
                draw, width, height,
                signals['strategy'],
                font_small
            )
        
        # Composite the overlay
        image = Image.alpha_composite(image, overlay)
        
        # Convert back to RGB for saving
        image = image.convert('RGB')
        
        # Convert to bytes
        output = io.BytesIO()
        image.save(output, format='PNG', quality=95)
        return output.getvalue()
    
    def _draw_trend_zones(self, draw, width, height, trend, regions):
        """Draw semi-transparent zones indicating trend areas"""
        chart_area = regions['chart_area']
        
        if trend == 'ALTA':
            # Draw bullish zone in lower half
            zone_y = chart_area['y'] + int(chart_area['height'] * 0.5)
            draw.rectangle(
                [
                    chart_area['x'],
                    zone_y,
                    chart_area['x'] + chart_area['width'],
                    chart_area['y'] + chart_area['height']
                ],
                fill=(0, 255, 0, 30)  # Light green
            )
        elif trend == 'BAIXA':
            # Draw bearish zone in upper half
            zone_y = chart_area['y'] + int(chart_area['height'] * 0.5)
            draw.rectangle(
                [
                    chart_area['x'],
                    chart_area['y'],
                    chart_area['x'] + chart_area['width'],
                    zone_y
                ],
                fill=(255, 0, 0, 30)  # Light red
            )
    
    def _draw_support_resistance_lines(self, draw, width, height, support_levels, resistance_levels, font):
        """Draw horizontal lines for support and resistance with labels"""
        # Draw support levels
        for i, level in enumerate(support_levels[:3]):
            y = height - 100 - (i * 60)  # Position from bottom
            
            # Draw dashed line
            self._draw_dashed_line(
                draw, 
                (50, y), 
                (width - 50, y), 
                self.colors['support'], 
                width=3
            )
            
            # Draw label
            label = f"Suporte: {level:.4f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            
            padding = 5
            label_x = width - text_width - 70
            draw.rectangle(
                [label_x - padding, y - 15, label_x + text_width + padding, y + 5],
                fill=(0, 0, 0, 200)
            )
            draw.text((label_x, y - 12), label, fill=self.colors['support'], font=font)
        
        # Draw resistance levels
        for i, level in enumerate(resistance_levels[:3]):
            y = 100 + (i * 60)  # Position from top
            
            # Draw dashed line
            self._draw_dashed_line(
                draw,
                (50, y),
                (width - 50, y),
                self.colors['resistance'],
                width=3
            )
            
            # Draw label
            label = f"Resistência: {level:.4f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            
            padding = 5
            label_x = width - text_width - 70
            draw.rectangle(
                [label_x - padding, y - 15, label_x + text_width + padding, y + 5],
                fill=(0, 0, 0, 200)
            )
            draw.text((label_x, y - 12), label, fill=self.colors['resistance'], font=font)
    
    def _draw_dashed_line(self, draw, start, end, color, width=2, dash_length=10):
        """Draw a dashed line"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate distance and angle
        dx = x2 - x1
        dy = y2 - y1
        distance = (dx**2 + dy**2)**0.5
        
        # Number of dashes
        num_dashes = int(distance / (dash_length * 2))
        
        for i in range(num_dashes):
            t1 = i * 2 * dash_length / distance
            t2 = (i * 2 + 1) * dash_length / distance
            
            if t2 > 1:
                t2 = 1
            
            px1 = x1 + t1 * dx
            py1 = y1 + t1 * dy
            px2 = x1 + t2 * dx
            py2 = y1 + t2 * dy
            
            draw.line([(px1, py1), (px2, py2)], fill=color, width=width)
    
    def _draw_trend_lines(self, draw, width, height, trend):
        """Draw trend lines indicating market direction"""
        if trend == 'ALTA':
            # Upward trend line
            points = [
                (width * 0.2, height * 0.7),
                (width * 0.8, height * 0.3)
            ]
            draw.line(points, fill=self.colors['trend_line'], width=4)
            
            # Add arrow at the end
            self._draw_arrow(draw, points[1], 'up', self.colors['trend_line'])
            
        elif trend == 'BAIXA':
            # Downward trend line
            points = [
                (width * 0.2, height * 0.3),
                (width * 0.8, height * 0.7)
            ]
            draw.line(points, fill=self.colors['trend_line'], width=4)
            
            # Add arrow at the end
            self._draw_arrow(draw, points[1], 'down', self.colors['trend_line'])
    
    def _draw_arrow(self, draw, position, direction, color, size=20):
        """Draw an arrow"""
        x, y = position
        
        if direction == 'up':
            points = [
                (x, y - size),
                (x - size/2, y),
                (x + size/2, y)
            ]
        elif direction == 'down':
            points = [
                (x, y + size),
                (x - size/2, y),
                (x + size/2, y)
            ]
        elif direction == 'left':
            points = [
                (x - size, y),
                (x, y - size/2),
                (x, y + size/2)
            ]
        elif direction == 'right':
            points = [
                (x + size, y),
                (x, y - size/2),
                (x, y + size/2)
            ]
        
        draw.polygon(points, fill=color)
    
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
        
        # Draw background with gradient effect (simulate with multiple rectangles)
        color = self.colors['call'] if action == 'CALL' else self.colors['put']
        padding = 15
        
        # Draw shadow
        draw.rectangle(
            [x - padding + 3, y - padding + 3, x + text_width + padding + 3, y + text_height + padding + 3],
            fill=(0, 0, 0, 150)
        )
        
        # Draw background
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 220)
        )
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            outline=color,
            width=4
        )
        
        # Draw text
        draw.text((x, y), text, fill=color, font=font)
    
    def _draw_entry_annotation_v2(self, draw, width, height, label, color, position, font):
        """Draw entry point annotation with improved positioning"""
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position based on entry type
        if position == 'bottom-left':
            x = width // 4
            y = height - 150
            arrow_direction = 'up'
            arrow_y = y - 40
        elif position == 'top-left':
            x = width // 4
            y = 120
            arrow_direction = 'down'
            arrow_y = y + text_height + 40
        elif position == 'bottom-right':
            x = width * 3 // 4 - text_width
            y = height - 150
            arrow_direction = 'up'
            arrow_y = y - 40
        else:  # top-right
            x = width * 3 // 4 - text_width
            y = 120
            arrow_direction = 'down'
            arrow_y = y + text_height + 40
        
        # Draw background box with shadow
        padding = 12
        draw.rectangle(
            [x - padding + 2, y - padding + 2, x + text_width + padding + 2, y + text_height + padding + 2],
            fill=(0, 0, 0, 150)
        )
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 200),
            outline=color,
            width=3
        )
        
        # Draw text
        draw.text((x, y), label, fill=color, font=font)
        
        # Draw arrow
        arrow_x = x + text_width // 2
        self._draw_arrow(draw, (arrow_x, arrow_y), arrow_direction, color, size=25)
    
    def _draw_exit_annotation(self, draw, width, height, label, color, font):
        """Draw take profit annotation"""
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = width - text_width - 80
        y = height // 3
        
        padding = 8
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180),
            outline=color,
            width=2
        )
        draw.text((x, y), label, fill=color, font=font)
    
    def _draw_stop_loss_annotation(self, draw, width, height, label, color, font):
        """Draw stop loss annotation"""
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = width - text_width - 80
        y = height * 2 // 3
        
        padding = 8
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180),
            outline=color,
            width=2
        )
        draw.text((x, y), label, fill=color, font=font)
    
    def _draw_strategy_label(self, draw, width, height, strategy, font):
        """Draw strategy type label"""
        label = f"📊 {strategy}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = 20
        y = height - 60
        
        # Draw background
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180),
            outline='#ffffff',
            width=2
        )
        
        # Draw text
        draw.text((x, y), label, fill='#ffffff', font=font)


def create_annotated_chart(image_bytes: bytes, analysis_text: str) -> bytes:
    """
    Convenience function to create annotated chart
    """
    annotator = ChartAnnotator()
    return annotator.annotate_chart(image_bytes, analysis_text)
