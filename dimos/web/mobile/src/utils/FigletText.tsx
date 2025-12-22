import React, {useMemo} from 'react';
import {Text, TextStyle, Platform} from 'react-native';
import figlet from 'figlet';
import ansiShadowFont from 'figlet/importable-fonts/ANSI Shadow';

figlet.parseFont('ANSI Shadow', ansiShadowFont);

interface FigletTextProps {
  text: string;
  font?: string;
  color?: string;
  fontSize?: number;
  style?: TextStyle;
  className?: string;
}

const FigletText: React.FC<FigletTextProps> = ({
  text,
  font = 'ANSI Shadow',
  color = '#FFF200',
  fontSize = 6,
  style = {},
  className = '',
}) => {
  const rendered = useMemo(() => {
    return figlet.textSync(text, {
      font: font as any,
      horizontalLayout: 'full',
      verticalLayout: 'default',
      width: 200,
      whitespaceBreak: true,
    });
  }, [text, font]);

  const monoFamily = Platform.select({
    ios: 'Menlo',
    android: 'monospace',
    default: 'monospace',
  });

  const composedStyle: TextStyle = {
    fontFamily: monoFamily,
    fontSize: fontSize,
    lineHeight: Math.ceil(fontSize * 1.15),
    color,
    fontWeight: '400',
    textAlign: 'left',
    letterSpacing: 0,
    ...(Platform.OS === 'android' && {includeFontPadding: false}),
    ...style,
  };

  return (
    <Text
      style={composedStyle}
      className={className}
      allowFontScaling={false}
    >
      {rendered}
    </Text>
  );
};

export default FigletText;
